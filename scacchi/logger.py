"""Training logger with tqdm progress bar and optional W&B backend."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from scacchi.config import LoggingConfig


class Logger:
    """
    Base logger with no-op backend — supports only tqdm progress bar updates.

    Subclass or use WandbLogger for a real logging backend.
    """

    def __init__(self, log_every: int = 1, max_steps: Optional[int] = None):
        self.log_every = log_every
        self.max_steps = max_steps
        self._initialized = False

    def should_log(self, step: int) -> bool:
        if self.log_every <= 0:
            return False
        is_periodic = step % self.log_every == 0
        is_last = self.max_steps is not None and step == self.max_steps - 1
        return is_periodic or is_last

    def __enter__(self):
        self._initialized = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._initialized = False
        return False

    def log_metrics(self, *args, **kwargs):
        pass

    def log_image(self, *args, **kwargs):
        pass

    def log_video(self, *args, **kwargs):
        pass

    def log(
        self,
        step: int,
        metrics: Any,
        pbar: Any = None,
        prefix: str = "train/",
        float_fmt: str = ".4f",
        pbar_filter: Optional[str] = None,
        **extra: Any,
    ):
        if not self.should_log(step):
            return

        raw = dict(metrics._asdict()) if hasattr(metrics, "_asdict") else dict(metrics)
        raw.update(extra)
        clean = self._convert_metrics(raw)

        if pbar is not None:
            self._update_pbar(pbar, clean, float_fmt, pbar_filter)

        self.log_metrics(step, clean, prefix)

    def _convert_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        clean: Dict[str, Any] = {}
        for k, v in metrics.items():
            if hasattr(v, "item"):
                v = v.item()
            if isinstance(v, (float, int)):
                clean[k] = v
            else:
                try:
                    clean[k] = float(v)
                except (ValueError, TypeError):
                    clean[k] = v
        return clean

    def _update_pbar(
        self,
        pbar: Any,
        metrics: Dict[str, Any],
        float_fmt: str,
        pbar_filter: Optional[str],
    ):
        filtered = metrics
        if pbar_filter is not None:
            pattern = re.compile(pbar_filter)
            filtered = {k: v for k, v in metrics.items() if pattern.search(k)}

        postfix = {}
        for k, v in filtered.items():
            postfix[k] = f"{v:{float_fmt}}" if isinstance(v, float) else str(v)

        pbar.set_postfix(**postfix)


class WandbLogger(Logger):
    def __init__(
        self,
        logging_cfg: LoggingConfig,
        log_every: int = 1,
        max_steps: Optional[int] = None,
        config: Any = None,
        dir: Optional[str] = None,
    ):
        super().__init__(log_every=log_every, max_steps=max_steps)
        self.project = logging_cfg.wandb_project
        self.config = config
        self.dir = dir
        self._run = None

    def __enter__(self):
        import wandb

        self._run = wandb.init(
            project=self.project,
            config=self.config,
            dir=self.dir,
            save_code=True,
        )
        self._initialized = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._run is not None:
            import wandb

            wandb.finish()
        self._initialized = False
        return False

    def log_metrics(self, step: int, metrics: Dict[str, Any], prefix: str = "train/"):
        import wandb

        wandb.log({f"{prefix}{k}": v for k, v in metrics.items()}, step=step)

    def log_image(
        self,
        step: int,
        key: str,
        image: Any,
        caption: Optional[str] = None,
    ):
        import numpy as np
        import wandb

        if hasattr(image, "__array__"):
            image = np.asarray(image)
        wandb.log({key: wandb.Image(image, caption=caption)}, step=step)

    def log_video(
        self,
        step: int,
        key: str,
        video_path: Path,
        format: Optional[Literal["gif", "mp4", "webm", "ogg"]] = "mp4",
        **kwargs,
    ):
        import wandb

        wandb.log({key: wandb.Video(str(video_path), format=format, **kwargs)}, step=step)


def build_logger(
    logging_cfg: LoggingConfig,
    log_every: int = 1,
    max_steps: Optional[int] = None,
    config: Any = None,
    dir: Optional[str] = None,
) -> Logger:
    if logging_cfg.wandb_enabled:
        return WandbLogger(
            logging_cfg,
            log_every=log_every,
            max_steps=max_steps,
            config=config,
            dir=dir,
        )
    return Logger(log_every=log_every, max_steps=max_steps)
