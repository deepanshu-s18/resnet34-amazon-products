"""Reproducibility utilities for deterministic experiment execution.

This module provides the seed-control infrastructure that every experiment
must call before constructing any tensors, datasets, or models.  Without it,
two runs with the same configuration will produce different results, making
ablation comparisons scientifically invalid.

Seeding scope
-------------
Five independent random number generators exist in the typical PyTorch
training stack.  All five must be seeded:

1. **Python built-in** ``random`` — used internally by Python and by some
   data-augmentation libraries.
2. **NumPy** — used by dataset splitting, WeightedRandomSampler, and many
   scientific libraries.
3. **PyTorch CPU** — controls all CPU tensor operations.
4. **PyTorch CUDA** (all devices) — controls GPU kernels.
5. **PYTHONHASHSEED** — controls Python's built-in hash randomisation
   (affects set/dict ordering in Python 3.3+).

DataLoader workers
------------------
Each DataLoader worker process *inherits* the parent RNG state at fork time.
Without :func:`make_worker_init_fn`, all workers share the same state and
produce correlated augmentation sequences — the mini-batch diversity is
illusory.  :func:`make_worker_init_fn` assigns each worker a unique seed:
``base_seed + worker_id``.

CUDA determinism
----------------
Setting ``torch.backends.cudnn.deterministic = True`` forces CUDA to use
deterministic algorithms.  This reduces throughput (typical overhead: 2–10 %)
but is required for bit-reproducible training.  Setting
``torch.use_deterministic_algorithms(True)`` additionally covers operations
outside cuDNN (e.g. scatter, index_put) and requires the environment variable
``CUBLAS_WORKSPACE_CONFIG=:4096:8``.

Checkpoint resumption
---------------------
:func:`get_rng_state` and :func:`restore_rng_state` capture and restore the
complete state of all RNGs.  Including these in every checkpoint means that
a resumed run is bit-for-bit identical to an uninterrupted run — the only
requirement for a resumed experiment to remain scientifically valid.

Example::

    from src.utils.seed import set_global_seed, make_worker_init_fn

    set_global_seed(42, deterministic=True)

    loader = DataLoader(
        dataset,
        worker_init_fn=make_worker_init_fn(42),
        generator=torch.Generator().manual_seed(42),
    )
"""

from __future__ import annotations

import logging
import os
import random
from typing import Callable, Optional

import numpy as np
import torch

__all__ = [
    "set_global_seed",
    "get_rng_state",
    "restore_rng_state",
    "make_worker_init_fn",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primary seed function
# ---------------------------------------------------------------------------


def set_global_seed(
    seed: int,
    deterministic: bool = True,
    warn_only: bool = False,
) -> None:
    """Seed every RNG in the Python / NumPy / PyTorch stack.

    Must be called **before** any of the following:

    - Dataset or DataLoader construction (affects augmentation sequences).
    - Model instantiation (affects weight initialisation).
    - Optimizer construction (affects any stochastic optimiser state).

    Args:
        seed: Non-negative integer seed value.  Use the same value for the
            experiment-level seed and the :func:`make_worker_init_fn` base
            seed so that all components are coordinated.
        deterministic: If ``True``, configure PyTorch and cuDNN to use
            deterministic algorithms.  Reduces throughput by roughly 2–10 %
            on CUDA but is required for bit-reproducible results.  Also sets
            ``CUBLAS_WORKSPACE_CONFIG`` to suppress the CUDA workspace size
            warning raised by ``torch.use_deterministic_algorithms``.
        warn_only: Forwarded to ``torch.use_deterministic_algorithms``.
            If ``True``, operations without deterministic implementations
            emit a warning instead of raising :class:`RuntimeError`.  Use
            ``True`` during development when debugging, ``False`` for final
            reported experiments.

    Raises:
        ValueError: If ``seed`` is negative.

    Example::

        set_global_seed(42)                        # strict determinism
        set_global_seed(42, deterministic=False)   # faster, non-deterministic
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}.")

    # 1. Python built-in RNG
    random.seed(seed)

    # 2. NumPy global RNG
    np.random.seed(seed)

    # 3. PyTorch CPU RNG
    torch.manual_seed(seed)

    # 4. PyTorch CUDA RNG — seed every visible GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 5. Python hash randomisation
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # cuDNN determinism: disables non-deterministic CUDA algorithms.
        torch.backends.cudnn.deterministic = True
        # Disable cuDNN benchmark mode: benchmark caches the fastest algorithm
        # for a given input size, but the selected algorithm may vary between
        # runs due to hardware state differences.
        torch.backends.cudnn.benchmark = False

        # Broader determinism: covers operations outside cuDNN (scatter,
        # index_put, etc.).  Requires CUBLAS_WORKSPACE_CONFIG when CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=warn_only)
            except TypeError:
                # PyTorch < 1.11 does not accept warn_only.
                torch.use_deterministic_algorithms(True)
    else:
        # benchmark=True lets cuDNN pick the fastest algorithm for each
        # input size at the cost of reproducibility.  Use during development
        # when iteration speed matters more than exact reproducibility.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    logger.debug(
        "Global seed set: seed=%d, deterministic=%s, warn_only=%s",
        seed,
        deterministic,
        warn_only,
    )


# ---------------------------------------------------------------------------
# RNG state capture and restoration
# ---------------------------------------------------------------------------


def get_rng_state() -> dict:
    """Capture the current state of all RNGs.

    The returned dict is suitable for inclusion in a training checkpoint
    (via :mod:`src.utils.checkpoint`).  Restoring it with
    :func:`restore_rng_state` makes a resumed training run bit-for-bit
    identical to an uninterrupted run.

    Returns:
        A dictionary with the following keys:

        - ``"python"``: :func:`random.getstate` output.
        - ``"numpy"``: :func:`numpy.random.get_state` output.
        - ``"torch"``: :func:`torch.get_rng_state` output.
        - ``"cuda"``: list of per-device CUDA RNG states, or ``None`` if
          CUDA is unavailable.

    Example::

        state = get_rng_state()
        # ... training step ...
        restore_rng_state(state)   # rewind to saved state
    """
    state: dict = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }
    return state


def restore_rng_state(state: dict) -> None:
    """Restore all RNGs to a previously captured state.

    Args:
        state: A dict returned by :func:`get_rng_state`.  Missing keys are
            silently ignored to allow loading checkpoints from a different
            hardware configuration (e.g. a CPU checkpoint resumed on GPU).

    Example::

        checkpoint = torch.load("checkpoint.pt")
        restore_rng_state(checkpoint["rng_states"])
    """
    if "python" in state:
        random.setstate(state["python"])

    if "numpy" in state:
        np.random.set_state(state["numpy"])

    if "torch" in state:
        torch.set_rng_state(state["torch"])

    if "cuda" in state and state["cuda"] is not None:
        if torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(state["cuda"])
            except RuntimeError as exc:
                # May fail when the number of GPUs differs between the
                # checkpoint machine and the current machine.
                logger.warning(
                    "Could not restore CUDA RNG state: %s. "
                    "Training will continue but is no longer bit-reproducible.",
                    exc,
                )

    logger.debug("RNG state restored from checkpoint.")


# ---------------------------------------------------------------------------
# DataLoader worker seeding
# ---------------------------------------------------------------------------


def make_worker_init_fn(base_seed: int) -> Callable[[int], None]:
    """Return a ``worker_init_fn`` that seeds each DataLoader worker uniquely.

    Each worker receives the seed ``base_seed + worker_id``.  This ensures:

    1. Workers produce *different* augmentation sequences (not correlated).
    2. The augmentation sequence is *reproducible* given the same
       ``base_seed`` — a different seed produces a different sequence.

    Pass the returned callable as the ``worker_init_fn`` argument of
    :class:`~torch.utils.data.DataLoader`::

        loader = DataLoader(
            dataset,
            num_workers=4,
            worker_init_fn=make_worker_init_fn(42),
            generator=torch.Generator().manual_seed(42),
        )

    Args:
        base_seed: The experiment-level seed.  Should match the seed passed
            to :func:`set_global_seed`.

    Returns:
        A callable ``(worker_id: int) -> None`` that seeds Python, NumPy,
        and PyTorch in each worker process.
    """

    def _init(worker_id: int) -> None:
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _init
