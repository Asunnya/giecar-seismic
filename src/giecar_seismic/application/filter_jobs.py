import threading
from collections.abc import Callable
from typing import Protocol

import numpy as np

from giecar_seismic.application.butterworth_filter import apply_lowpass_filter
from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.domain.job import InvalidTransitionError, Job, JobStatus


class DatasetRepository(Protocol):
    def get(self, dataset_id: int) -> SeismicDataset | None: ...


class JobRepository(Protocol):
    def add(self, job: Job) -> Job: ...
    def get(self, job_id: int) -> Job | None: ...
    def update(self, job: Job) -> None: ...
    def list(
        self, dataset_id: int | None = None, status: JobStatus | None = None
    ) -> list[Job]: ...


class TraceReader(Protocol):
    trace_count: int

    def read_chunk(self, start: int, stop: int) -> np.ndarray: ...

    def close(self) -> None:
        """Release the underlying resource. Always safe to call once done."""
        ...


class TraceWriter(Protocol):
    def write_chunk(self, start: int, chunk: np.ndarray) -> None: ...

    def finalize(self) -> None:
        """Mark the output as semantically complete and valid.

        Called only on the success path, after every chunk has been
        written. Does NOT release the underlying resource -- close() is
        always called afterwards to do that. Never used as a generic
        cleanup hook.
        """
        ...

    def close(self) -> None:
        """Release the underlying resource.

        Always called once the writer was created, on every path. On the
        success path it runs after finalize() already marked the output
        complete; on any other path (error or cancellation) it is the only
        writer-side cleanup that runs.
        """
        ...


class CancelToken(Protocol):
    def is_cancelled(self) -> bool: ...

    def request_cancel(self) -> None:
        """Ask a worker observing this token to stop at its next safe
        boundary.

        Minimal addition to the original is_cancelled()-only contract:
        FilterJobService.cancel_job() does not receive the token that a
        given run_filter_job() call is polling (see CancelToken/CancelToken
        registration below), so it needs a way to signal *that specific*
        token instance instead of mutating Job state directly. Calling
        this more than once must be safe and have no additional effect.

        MUST be thread-safe, non-blocking, and O(1) (no I/O, no waiting on
        another lock): FilterJobService.cancel_job() calls this while
        still holding its internal registration lock, precisely to make
        "accept the request" and "make it visible to the worker" a single
        atomic step. A slow or blocking implementation would hold that
        lock for longer than the cheap, in-memory bookkeeping it protects.
        """
        ...


class CooperativeCancelToken:
    """Thread-safe CancelToken backed by threading.Event.

    Meant to be created by whoever starts a job (the future Qt GUI
    thread), handed to run_filter_job() (running on a future QThread
    worker), and signalled via request_cancel() from FilterJobService.
    cancel_job() -- possibly called from yet another thread. Event.set()
    and Event.is_set() are documented as safe to call concurrently from
    different threads; that guarantee is deliberately not left to depend
    on CPython's GIL.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def request_cancel(self) -> None:
        self._event.set()


ReaderFactory = Callable[[SeismicDataset], TraceReader]
WriterFactory = Callable[[Job], TraceWriter]
ProgressCallback = Callable[[float], None]


class DatasetNotFoundError(Exception):
    pass


class JobNotFoundError(Exception):
    pass


class InvalidFilterParametersError(ValueError):
    pass


class JobAlreadyRunningError(Exception):
    """Raised by run_filter_job() when the given job id already has an
    active execution. At most one worker may own a given job id at a
    time; a second concurrent attempt is rejected outright rather than
    overwriting or removing the legitimate run's token registration.
    """


class CancellationWindowClosedError(Exception):
    """Raised by cancel_job() when the job's worker has already committed
    to finalizing its output (crossed the point of no return) and can no
    longer honor a cancellation request. See _RunRegistration.committed.
    """


class NoActiveExecutionError(Exception):
    """Raised by cancel_job() when the job is RUNNING but no
    run_filter_job() execution is registered for it (e.g. the Job was
    forced into RUNNING directly, bypassing run_filter_job(), or the
    worker already finished and deregistered). There is nobody to signal,
    so the request is rejected explicitly rather than reported as if it
    had been accepted.
    """


class _RunRegistration:
    """Tracks the single active run_filter_job() execution for one job id.

    `committed` is the point-of-no-return flag: it starts False, and the
    worker sets it to True -- atomically, under the same lock cancel_job()
    reads it with -- the instant it decides to proceed to finalize()
    instead of cancelling. Once True, cancel_job() must reject the
    request explicitly instead of accepting one it can no longer honor.
    """

    __slots__ = ("committed", "token")

    def __init__(self, token: CancelToken) -> None:
        self.token = token
        self.committed = False


MIN_FILTER_ORDER = 2
MAX_FILTER_ORDER = 8

# 256 traces/chunk: notebooks/01_inspect_segy.ipynb's exploratory benchmark
# found this in the flat/best region for the sample survey. Not a proof of
# optimality (page-cache effects), just a reasonable default.
DEFAULT_CHUNK_SIZE = 256


class FilterJobService:
    def __init__(
        self,
        datasets: DatasetRepository,
        jobs: JobRepository,
        reader_factory: ReaderFactory | None = None,
        writer_factory: WriterFactory | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ):
        self._datasets = datasets
        self._jobs = jobs
        self._reader_factory = reader_factory
        self._writer_factory = writer_factory
        self._chunk_size = chunk_size
        # Tracks the single active run_filter_job() execution per job id
        # (see _RunRegistration), so that:
        #   - cancel_job() -- which does not receive the token as an
        #     argument -- can still find and signal the token a given
        #     run_filter_job() call is polling;
        #   - at most one execution can be registered for a job id at a
        #     time, so a second concurrent run_filter_job() call for the
        #     same job cannot overwrite or remove the legitimate one's
        #     registration.
        # Guarded by a lock because cancel_job() (e.g. the Qt GUI thread)
        # and run_filter_job() (e.g. a QThread worker) run concurrently on
        # different threads; a plain dict has no defined behaviour under
        # concurrent read/write and must not be relied on for this. Only
        # cheap, in-memory operations ever happen while holding it -- never
        # I/O such as reading/filtering/writing a chunk or finalize().
        self._cancel_tokens: dict[int, _RunRegistration] = {}
        self._cancel_tokens_lock = threading.Lock()

    def create_filter_job(self, dataset_id: int, cutoff_hz: float, order: int) -> Job:
        dataset = self._datasets.get(dataset_id)
        if dataset is None:
            raise DatasetNotFoundError(f"dataset {dataset_id} not found")

        if not (0 < cutoff_hz < dataset.nyquist_hz):
            raise InvalidFilterParametersError(
                f"cutoff_hz must be between 0 and the dataset's Nyquist "
                f"frequency ({dataset.nyquist_hz} Hz), got {cutoff_hz}"
            )

        if not (MIN_FILTER_ORDER <= order <= MAX_FILTER_ORDER):
            raise InvalidFilterParametersError(
                f"order must be between {MIN_FILTER_ORDER} and "
                f"{MAX_FILTER_ORDER}, got {order}"
            )

        job = Job(dataset_id=dataset_id, cutoff_hz=cutoff_hz, order=order)
        return self._jobs.add(job)

    def list_jobs(
        self, dataset_id: int | None = None, status: JobStatus | None = None
    ) -> list[Job]:
        return self._jobs.list(dataset_id=dataset_id, status=status)

    def get_job_status(self, job_id: int) -> Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(f"job {job_id} not found")
        return job

    def cancel_job(self, job_id: int) -> Job:
        # cancel_job() represents only the *request* to cancel -- it must
        # never itself perform RUNNING -> CANCELLED. Only run_filter_job()
        # (the worker) may do that, after observing the request at a safe
        # chunk boundary; see the two is_cancelled() checks there. This
        # keeps a single writer of that transition instead of two racing
        # sources (the caller and the worker).
        job = self.get_job_status(job_id)
        if job.status is not JobStatus.RUNNING:
            raise InvalidTransitionError(f"cannot cancel a job in status {job.status}")

        with self._cancel_tokens_lock:
            registration = self._cancel_tokens.get(job_id)
            if registration is None:
                # RUNNING but nobody registered to receive the request --
                # e.g. the Job was forced into RUNNING directly, bypassing
                # run_filter_job(), or the worker already finished and
                # deregistered. Reporting success here would be a lie:
                # there is no worker left that will ever look at a token
                # again, so the request is rejected explicitly instead of
                # silently accepted and then ignored.
                raise NoActiveExecutionError(
                    f"job {job_id} is RUNNING but has no active "
                    "run_filter_job() execution registered to receive a "
                    "cancellation request"
                )
            if registration.committed:
                # The worker has already decided, under this same lock, to
                # proceed straight to finalize() -- it will never look at
                # cancel_token again. Promising cancellation now would be
                # a lie the caller could act on (e.g. a GUI showing
                # "cancelling..." for a job that is about to complete), so
                # this is rejected explicitly instead of being silently
                # accepted and then ignored.
                raise CancellationWindowClosedError(
                    f"job {job_id} has already started finalizing its "
                    "output and can no longer be cancelled"
                )
            # Linearization point for cancellation: request_cancel() runs
            # here, still holding the very same lock the worker's
            # point-of-no-return check uses (see run_filter_job()). This
            # makes "accept the request" (the committed check above) and
            # "make it visible to the worker" a single atomic step --
            # closing the race where releasing the lock first and calling
            # request_cancel() as a separate, unsynchronized step could
            # let the worker acquire the lock, observe the token as not
            # yet cancelled, and commit to finalize() before the request
            # ever became visible. This is safe only because
            # CancelToken.request_cancel() is required to be O(1),
            # non-blocking, and thread-safe (see its Protocol docstring);
            # CooperativeCancelToken.request_cancel() is just
            # threading.Event.set() -- idempotent, so two cancel_job()
            # calls in a row (or one racing a worker about to finish) is
            # safe too.
            registration.token.request_cancel()

        return job

    def run_filter_job(
        self,
        job_id: int,
        progress_callback: ProgressCallback,
        cancel_token: CancelToken,
    ) -> Job:
        if self._reader_factory is None or self._writer_factory is None:
            raise RuntimeError(
                "FilterJobService requires reader_factory and writer_factory "
                "to run a filter job"
            )

        # Claim exclusive ownership of this job id *before* doing anything
        # else -- including fetching the Job/dataset. The check ("is
        # anyone already registered for this job id?") and the write
        # ("register me") happen atomically under one lock acquisition, so
        # two concurrent run_filter_job() calls for the same job_id can
        # never both succeed: exactly one registers and proceeds, the
        # other is rejected immediately, before it ever touches the
        # reader/writer factories or the job's state. This also means
        # cancel_job() finds a token registered as soon as the job
        # becomes visible as RUNNING (job.start() runs strictly after
        # this), closing that race too.
        with self._cancel_tokens_lock:
            if job_id in self._cancel_tokens:
                raise JobAlreadyRunningError(
                    f"job {job_id} already has an active run_filter_job() "
                    "execution"
                )
            registration = _RunRegistration(cancel_token)
            self._cancel_tokens[job_id] = registration

        try:
            job = self.get_job_status(job_id)
            dataset = self._datasets.get(job.dataset_id)
            if dataset is None:
                raise DatasetNotFoundError(f"dataset {job.dataset_id} not found")

            job.start()
            self._jobs.update(job)

            reader: TraceReader | None = None
            writer: TraceWriter | None = None

            try:
                # Both factories run inside the try block: once the job is
                # RUNNING, any failure -- including opening the
                # reader/writer themselves -- must still resolve to a
                # terminal job state and never leak whichever of the two
                # resources did get created.
                reader = self._reader_factory(dataset)
                writer = self._writer_factory(job)

                trace_count = reader.trace_count
                for start in range(0, trace_count, self._chunk_size):
                    if cancel_token.is_cancelled():
                        job.cancel()
                        return job

                    stop = min(start + self._chunk_size, trace_count)
                    chunk = reader.read_chunk(start, stop)
                    filtered = apply_lowpass_filter(
                        chunk, job.cutoff_hz, job.order, dataset.sample_rate_ms
                    )
                    writer.write_chunk(start, filtered)
                    progress_callback(round(100 * stop / trace_count))

                # Point of no return. The per-iteration check above only
                # runs *before* a chunk -- there is no "next" chunk after
                # the last one, so without this check a cancellation
                # requested while the final chunk was being processed
                # would never be observed. The check-and-decide step
                # itself is atomic with cancel_job()'s own critical
                # section (same lock), so exactly one of two things
                # happens for any cancel_job() call made around this
                # point: either it lands *before* this section runs, and
                # is_cancelled() below observes it (-> CANCELLED); or it
                # lands *after* this section already set `committed`, and
                # cancel_job() rejects it via CancellationWindowClosedError
                # instead of silently accepting a promise it cannot keep.
                # Only the flag read/write happens under the lock --
                # finalize() itself (the actual I/O) runs after it is
                # released.
                with self._cancel_tokens_lock:
                    cancelled = cancel_token.is_cancelled()
                    if not cancelled:
                        registration.committed = True

                if cancelled:
                    job.cancel()
                    return job

                writer.finalize()
                job.complete()
            except Exception as exc:  # noqa: BLE001 -- any failure must FAIL the job, not crash the worker
                job.fail(str(exc))
            finally:
                # Ownership: whoever created a resource here is
                # responsible for releasing it, on every path. finalize()
                # (success only, run above) and close() (unconditional
                # release, run here) are deliberately separate calls --
                # close() never substitutes for finalize() and is safe to
                # call whether or not finalize() ran or succeeded.
                #
                # Cleanup is best-effort and independent per resource: a
                # close() failure is swallowed here rather than raised, so
                # (a) the reader failing to close can never stop the
                # writer's own close() from being attempted, and (b) a
                # cleanup failure can never overwrite the primary failure
                # already recorded on the job (via job.fail() above) or
                # leave the job stuck in RUNNING. Not re-raised or logged
                # yet -- that's future work, not part of this lifecycle
                # fix.
                if reader is not None:
                    try:
                        reader.close()
                    except Exception:  # noqa: BLE001, S110 -- best-effort cleanup, see comment above
                        pass
                if writer is not None:
                    try:
                        writer.close()
                    except Exception:  # noqa: BLE001, S110 -- best-effort cleanup, see comment above
                        pass
                self._jobs.update(job)
        finally:
            # Release ownership of this job id. Because registration above
            # is exclusive (a second concurrent run_filter_job() call for
            # the same job_id raises JobAlreadyRunningError before ever
            # reaching this try/finally), this call is always popping its
            # own registration -- never one belonging to another run. This
            # also lets a later cancel_job() call -- after this job
            # reached a terminal state, or before a future run under the
            # same job id -- never signal a stale token.
            with self._cancel_tokens_lock:
                self._cancel_tokens.pop(job_id, None)

        return job
