import os

import pytest

# Deterministic CUDA tests need a fixed cuBLAS workspace; set it before torch starts CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@pytest.fixture(autouse=True)
def _restore_tf32():
    """Training entry points enable TF32 globally; keep that from leaking into later tests."""

    try:
        import torch
    except ImportError:
        yield
        return
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved
