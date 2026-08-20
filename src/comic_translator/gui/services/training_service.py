"""Training Service — 包装 scripts/train_bubble_model.py，提供流式训练日志

不复制训练逻辑，只是 subprocess 调用 + 输出解析 + 状态跟踪。
"""

from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

from loguru import logger


@dataclass
class TrainingStatus:
    """训练任务状态"""
    running: bool = False
    return_code: int | None = None
    dataset: str = ""
    model: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 0
    latest_metrics: dict = field(default_factory=dict)
    output_path: str = ""
    error: str = ""


# 从 ultralytics 训练输出中解析 epoch / loss / mAP
EPOCH_RE = re.compile(r"^\s*(\d+)/(\d+)\s")
MAP_RE = re.compile(r"all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")


class TrainingService:
    """训练任务管理（单例，单任务）"""

    def __init__(self, script_path: str = "scripts/train_bubble_model.py"):
        self.script_path = Path(script_path)
        self._process: subprocess.Popen | None = None
        self._status = TrainingStatus()
        self._log_buffer: list[str] = []
        self._lock = threading.Lock()

    # ---------- 状态 ----------

    @property
    def status(self) -> TrainingStatus:
        with self._lock:
            return self._status

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    # ---------- 启动 ----------

    def start_training(
        self,
        dataset_yaml: str,
        base_model: str = "models/bubble/yolo11n-seg.pt",
        epochs: int = 50,
        batch: int = 4,
        imgsz: int = 1600,
        device: str = "0",
        name: str = "bubble",
    ) -> bool:
        """启动训练子进程；返回是否成功启动"""
        if self.is_running():
            logger.warning("Training already running")
            return False

        if not Path(dataset_yaml).exists():
            raise FileNotFoundError(f"数据集配置不存在: {dataset_yaml}")
        if not self.script_path.exists():
            raise FileNotFoundError(f"训练脚本不存在: {self.script_path}")

        cmd = [
            sys.executable, str(self.script_path),
            "--data", dataset_yaml,
            "--model", base_model,
            "--epochs", str(epochs),
            "--batch", str(batch),
            "--imgsz", str(imgsz),
            "--device", device,
            "--name", name,
        ]

        with self._lock:
            self._status = TrainingStatus(
                running=True,
                dataset=dataset_yaml,
                model=base_model,
                started_at=time.time(),
                total_epochs=epochs,
                # 与 scripts/train_bubble_model.py 的导出路径保持一致
                output_path=f"models/bubble/{Path(base_model).stem}-{name}.pt",
            )
            self._log_buffer = [f"$ {' '.join(cmd)}"]

        logger.info(f"Starting training: {' '.join(cmd)}")
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        # 后台线程读输出
        threading.Thread(target=self._reader, daemon=True).start()
        return True

    # ---------- 取消 ----------

    def cancel(self) -> bool:
        if not self.is_running():
            return False
        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
        with self._lock:
            self._status.running = False
            self._status.error = "已取消"
            self._status.finished_at = time.time()
        return True

    # ---------- 流式日志 ----------

    def stream_logs(self, from_index: int = 0) -> tuple[list[str], int]:
        """返回 (新日志, 当前总数) — 调用方轮询"""
        with self._lock:
            new_lines = self._log_buffer[from_index:]
            return new_lines, len(self._log_buffer)

    def stream_training(
        self,
        poll_interval: float = 1.0,
    ) -> Generator[tuple[list[str], TrainingStatus], None, None]:
        """生成器：持续 yield (新增日志, 当前状态) 直到训练结束"""
        idx = 0
        while True:
            new_lines, idx = self.stream_logs(idx)
            status = self.status
            yield new_lines, status
            if not status.running and not new_lines:
                break
            time.sleep(poll_interval)

    # ---------- 内部 ----------

    def _reader(self):
        """后台线程：读取 subprocess stdout，更新状态"""
        try:
            for line in self._process.stdout:
                line = line.rstrip()
                if not line:
                    continue
                with self._lock:
                    self._log_buffer.append(line)
                    self._parse_line(line)
        except Exception as e:
            with self._lock:
                self._log_buffer.append(f"[reader error] {e}")
        finally:
            rc = self._process.wait()
            with self._lock:
                self._status.running = False
                self._status.return_code = rc
                self._status.finished_at = time.time()
                if rc != 0 and not self._status.error:
                    self._status.error = f"训练退出码 {rc}"

    def _parse_line(self, line: str):
        """从输出中解析 epoch 进度和 mAP"""
        m = EPOCH_RE.match(line)
        if m:
            self._status.current_epoch = int(m.group(1))
            self._status.total_epochs = int(m.group(2))

        m = MAP_RE.search(line)
        if m:
            self._status.latest_metrics = {
                "precision": float(m.group(1)),
                "recall": float(m.group(2)),
                "mAP50": float(m.group(3)),
                "mAP50-95": float(m.group(4)),
            }

        # 捕获 "mAP50:     0.xxxx" 这种最终汇总
        if "mAP50:" in line and "mAP50-95" not in line:
            try:
                val = float(line.split(":")[-1].strip())
                self._status.latest_metrics.setdefault("mAP50", val)
            except ValueError:
                pass
        elif "mAP50-95:" in line:
            try:
                val = float(line.split(":")[-1].strip())
                self._status.latest_metrics.setdefault("mAP50-95", val)
            except ValueError:
                pass
