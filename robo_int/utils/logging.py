"""轻量实验日志：console + jsonl 指标文件；wandb 可选。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


class Logger:
    def __init__(self, log_dir: str | Path, wandb_kwargs: dict | None = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.log_dir / "metrics.jsonl"
        self._wandb = None
        if wandb_kwargs is not None:
            try:
                import wandb

                self._wandb = wandb.init(**wandb_kwargs)
            except ImportError:
                print("[warn] 未安装 wandb，仅本地日志。pip install 'robo-int[wandb]'")

    def log(self, step: int, metrics: dict) -> None:
        record = {"step": step, "time": time.time(), **metrics}
        print(
            f"[step {step}] "
            + " ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()),
            file=sys.stderr,
        )
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
