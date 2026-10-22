import os

THREAD_CAP_VARIABLES = (
    "OMP_NUM_THREADS",          # OpenMP: FAISS, PyTorch, simlar's compiled kernels
    "OPENBLAS_NUM_THREADS",     # OpenBLAS: numpy and scipy, so BM25, TF-IDF, Qdrant local mode
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "RAYON_NUM_THREADS",        # Rust rayon: TurboVec, Chroma's core
    "TOKIO_WORKER_THREADS",     # Chroma's async runtime
)

#: Noise that would otherwise land in the middle of a timed span.
QUIET_VARIABLES = {
    "TQDM_DISABLE": "1",
    "GRPC_VERBOSITY": "ERROR",   # Milvus Lite's gRPC channel
    "GLOG_minloglevel": "2",
}


def available_cores() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def configure() -> int:
    for variable in THREAD_CAP_VARIABLES:
        os.environ.pop(variable, None)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ.update(QUIET_VARIABLES)
    return available_cores()
