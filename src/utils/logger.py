"""Unified experiment logging: structured Python logs and TensorBoard.

Every training run produces two complementary audit trails:

1. **Structured text logs** — timestamped, levelled messages written to both
   the console (``INFO`` and above) and a per-experiment log file (``DEBUG``
   and above).  These are the primary record of what happened during training
   and are the first place to look when a run fails.

2. **TensorBoard event files** — scalar curves, images, and histograms that
   can be explored interactively with ``tensorboard --logdir results/``.
   TensorBoard is the primary tool for comparing training runs and diagnosing
   optimisation problems.

:class:`ExperimentLogger` wraps both behind a single interface so that
training code logs once and both destinations receive the message.

Directory layout
----------------
Given ``log_dir = Path("results/cifar10_resnet34_seed42")``, the logger
creates::

    results/
    └── cifar10_resnet34_seed42/
        ├── experiment.log          ← full debug log (text)
        └── tensorboard/            ← TensorBoard event files
            └── events.out.tfevents.*

Example::

    from pathlib import Path
    from src.utils.logger import ExperimentLogger

    with ExperimentLogger("cifar10_resnet34", Path("results/run_001")) as log:
        log.info("Training started")
        for epoch in range(100):
            log.log_scalar("train/loss", loss, step=epoch)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional, Union

import torch

__all__ = ["ExperimentLogger", "get_logger", "setup_root_logging"]

# ---------------------------------------------------------------------------
# Module-level format constants
# ---------------------------------------------------------------------------

_CONSOLE_FORMAT = "[%(asctime)s][%(levelname)-8s] %(message)s"
_FILE_FORMAT    = "[%(asctime)s][%(levelname)-8s][%(name)s:%(lineno)d] %(message)s"
_DATE_FORMAT    = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# Root logging helper
# ---------------------------------------------------------------------------


def setup_root_logging(level: str = "INFO") -> None:
    """Configure the root Python logger once for the whole process.

    Call this at the entry point (``scripts/train.py``) before any other
    logging calls.  Subsequent calls are no-ops if the root logger already
    has handlers.

    Args:
        level: Minimum severity to show on the console.
            One of ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``.
    """
    root = logging.getLogger()
    if root.handlers:
        return   # Already configured — do not add duplicate handlers.

    numeric = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(logging.DEBUG)  # Capture everything; handlers filter.

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric)
    handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT, _DATE_FORMAT))
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (standard Python logger hierarchy).

    Prefer this over ``logging.getLogger`` directly so that all module
    loggers are documented as going through the same naming scheme::

        logger = get_logger(__name__)

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` instance.
    """
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# ExperimentLogger
# ---------------------------------------------------------------------------


class ExperimentLogger:
    """Unified interface for Python logging and TensorBoard.

    Wraps a :class:`logging.Logger` (console + file) and a
    :class:`~torch.utils.tensorboard.SummaryWriter` (TensorBoard) behind a
    single API.  All ``log_*`` methods write to both destinations.

    The class supports use as a context manager to ensure the TensorBoard
    writer is flushed and closed even if training raises an exception::

        with ExperimentLogger(name, log_dir) as log:
            trainer.train(logger=log)

    Alternatively, call :meth:`close` explicitly at the end of the script.

    Args:
        experiment_name: Human-readable identifier, e.g.
            ``"cifar10_resnet34_seed42"``.  Used as the Python logger name
            and embedded in the log file name.
        log_dir: Root directory for this experiment's outputs.  The logger
            creates ``log_dir/experiment.log`` and ``log_dir/tensorboard/``.
        level: Minimum log level for the console handler.
            Defaults to ``"INFO"``.  The file handler always captures
            ``DEBUG`` and above.
        use_tensorboard: If ``False``, the TensorBoard writer is not created.
            Useful for unit tests or environments without TensorBoard.

    Attributes:
        experiment_name: The experiment identifier.
        log_dir: The experiment output directory as a :class:`~pathlib.Path`.
    """

    def __init__(
        self,
        experiment_name: str,
        log_dir: Path,
        level: str = "INFO",
        use_tensorboard: bool = True,
    ) -> None:
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Python logger
        # ------------------------------------------------------------------
        self._logger = logging.getLogger(experiment_name)
        self._logger.setLevel(logging.DEBUG)
        # Prevent log records from propagating to the root logger when the
        # root already has a StreamHandler — avoids duplicate console output.
        self._logger.propagate = False

        numeric_level = getattr(logging, level.upper(), logging.INFO)

        # Console handler — INFO and above.
        if not any(isinstance(h, logging.StreamHandler) for h in self._logger.handlers):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(numeric_level)
            console_handler.setFormatter(
                logging.Formatter(_CONSOLE_FORMAT, _DATE_FORMAT)
            )
            self._logger.addHandler(console_handler)

        # File handler — DEBUG and above (full audit trail).
        log_file = self.log_dir / "experiment.log"
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(_FILE_FORMAT, _DATE_FORMAT)
        )
        self._logger.addHandler(file_handler)

        # ------------------------------------------------------------------
        # TensorBoard writer
        # ------------------------------------------------------------------
        self._writer: Optional[Any] = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter  # type: ignore

                tb_dir = self.log_dir / "tensorboard"
                tb_dir.mkdir(parents=True, exist_ok=True)
                self._writer = SummaryWriter(log_dir=str(tb_dir))
                self._logger.debug(
                    "TensorBoard writer initialised at %s", tb_dir
                )
            except ImportError:
                self._logger.warning(
                    "tensorboard is not installed — TensorBoard logging disabled. "
                    "Install with: pip install tensorboard"
                )

        self._logger.info(
            "ExperimentLogger ready | experiment=%s | log_dir=%s",
            experiment_name,
            log_dir,
        )

    # -----------------------------------------------------------------------
    # Context manager
    # -----------------------------------------------------------------------

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # -----------------------------------------------------------------------
    # Python logging pass-throughs
    # -----------------------------------------------------------------------

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a DEBUG-level message."""
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an INFO-level message."""
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a WARNING-level message."""
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an ERROR-level message."""
        self._logger.error(msg, *args, **kwargs)

    # -----------------------------------------------------------------------
    # TensorBoard logging
    # -----------------------------------------------------------------------

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a single scalar value to TensorBoard.

        Args:
            tag: Metric name, e.g. ``"train/loss"`` or ``"val/accuracy"``.
                Use ``/`` to group related scalars in the TensorBoard UI.
            value: The scalar value to log.
            step: Global step (or epoch) index.
        """
        if self._writer is not None:
            self._writer.add_scalar(tag, value, global_step=step)

    def log_scalars(
        self,
        tag_value_pairs: dict[str, float],
        step: int,
    ) -> None:
        """Log multiple scalars in one call.

        Args:
            tag_value_pairs: Mapping of metric name → value.
            step: Global step index.
        """
        for tag, value in tag_value_pairs.items():
            self.log_scalar(tag, value, step)

    def log_image(
        self,
        tag: str,
        image: torch.Tensor,
        step: int,
        dataformats: str = "CHW",
    ) -> None:
        """Log an image tensor to TensorBoard.

        Args:
            tag: Image group name, e.g. ``"grad_cam/sofa"``.
            image: Float tensor in ``[0, 1]`` with shape matching
                ``dataformats``.
            step: Global step index.
            dataformats: Axis order string, e.g. ``"CHW"`` or ``"HWC"``.
        """
        if self._writer is not None:
            self._writer.add_image(tag, image, global_step=step,
                                   dataformats=dataformats)

    def log_images(
        self,
        tag: str,
        images: torch.Tensor,
        step: int,
    ) -> None:
        """Log a batch of images to TensorBoard (shape ``NCHW``).

        Args:
            tag: Image group name.
            images: Float tensor of shape ``(N, C, H, W)`` in ``[0, 1]``.
            step: Global step index.
        """
        if self._writer is not None:
            self._writer.add_images(tag, images, global_step=step)

    def log_histogram(
        self,
        tag: str,
        tensor: torch.Tensor,
        step: int,
    ) -> None:
        """Log a histogram of tensor values to TensorBoard.

        Useful for monitoring weight distributions and gradient magnitudes
        over training.

        Args:
            tag: Histogram name, e.g. ``"weights/layer4.conv1"``.
            tensor: Tensor whose values are histogrammed.
            step: Global step index.
        """
        if self._writer is not None:
            self._writer.add_histogram(
                tag, tensor.detach().cpu().float(), global_step=step
            )

    def log_config(self, config: dict[str, Any], step: int = 0) -> None:
        """Serialise the experiment config as TensorBoard text.

        The config is also written to ``log_dir/config_summary.txt`` for
        easy inspection without TensorBoard.

        Args:
            config: Flat or nested dict of experiment hyperparameters.
            step: Step value for the TensorBoard record.
        """
        # Format as a readable YAML-like block.
        lines = ["**Experiment Configuration**\n\n```"]
        lines += [f"{k}: {v}" for k, v in _flatten_dict(config).items()]
        lines.append("```")
        text = "\n".join(lines)

        if self._writer is not None:
            self._writer.add_text("config", text, global_step=step)

        # Plain-text backup.
        config_txt = self.log_dir / "config_summary.txt"
        with open(config_txt, "w") as fh:
            fh.write("\n".join(
                [f"{k}: {v}" for k, v in _flatten_dict(config).items()]
            ))
        self._logger.debug("Config summary written to %s", config_txt)

    def log_model_graph(
        self,
        model: torch.nn.Module,
        sample_input: torch.Tensor,
    ) -> None:
        """Log the model computation graph to TensorBoard.

        Args:
            model: The model to graph.
            sample_input: A sample input tensor (batch of one is sufficient).
        """
        if self._writer is not None:
            try:
                self._writer.add_graph(model, sample_input)
                self._logger.debug("Model graph logged to TensorBoard.")
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "Failed to log model graph: %s. Continuing.", exc
                )

    def flush(self) -> None:
        """Flush the TensorBoard writer to disk.

        Call after every epoch to ensure data is visible in TensorBoard
        even if the process is interrupted.
        """
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        """Flush and close the TensorBoard writer.

        Must be called at the end of the experiment, or use the class as a
        context manager.  Failing to close may result in incomplete
        TensorBoard event files.
        """
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None
            self._logger.debug("TensorBoard writer closed.")

        # Remove file handlers to avoid logging to a stale file if the
        # logger object is reused (e.g. in test environments).
        for handler in list(self._logger.handlers):
            handler.flush()
            handler.close()
            self._logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _flatten_dict(
    d: dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    """Recursively flatten a nested dict for logging.

    Example::

        _flatten_dict({"a": {"b": 1}}) == {"a.b": 1}
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
