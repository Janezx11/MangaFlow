"""环境体检 —— doctor 命令的实现层

检查 Python 版本、依赖安装、模型文件、配置有效性与数据目录，
输出 ✅/⚠️/❌ 分级报告。纯检查，不修改任何状态。
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

OK = "OK"
WARN = "WARN"
FAIL = "FAIL"


@dataclass
class CheckResult:
    """单项检查结果"""

    status: str  # OK / WARN / FAIL
    category: str
    name: str
    detail: str = ""
    hint: str = ""


@dataclass
class DoctorReport:
    """体检汇总"""

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == FAIL]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == WARN]

    @property
    def ok(self) -> bool:
        return not self.failures

    def to_lines(self) -> list[str]:
        icon = {OK: "✅", WARN: "⚠️ ", FAIL: "❌"}
        lines = []
        last_cat = ""
        for c in self.checks:
            if c.category != last_cat:
                lines.append(f"— {c.category} —")
                last_cat = c.category
            line = f"{icon[c.status]} {c.name}"
            if c.detail:
                line += f"  {c.detail}"
            lines.append(line)
            if c.hint:
                lines.append(f"    → {c.hint}")
        return lines


def _check_python(report: DoctorReport) -> None:
    v = sys.version_info
    if (v.major, v.minor) >= (3, 10):
        report.checks.append(CheckResult(OK, "Python", f"{v.major}.{v.minor}.{v.micro}"))
    else:
        report.checks.append(CheckResult(
            FAIL, "Python", f"{v.major}.{v.minor}",
            hint="需要 Python >= 3.10",
        ))


def _check_import(report: DoctorReport, category: str, module: str, *,
                  extra: str = "", required: bool = True, pip_name: str = "") -> None:
    try:
        importlib.import_module(module)
        report.checks.append(CheckResult(OK, category, module))
    except Exception as e:
        name = pip_name or module
        if required:
            report.checks.append(CheckResult(
                FAIL, category, module, detail=f"导入失败（{e.__class__.__name__}）",
                hint=f"pip install {extra} {name}".strip(),
            ))
        else:
            report.checks.append(CheckResult(
                WARN, category, module, detail="未安装（可选扩展）",
                hint=f"pip install -e \".[{extra}]\"" if extra else f"pip install {name}",
            ))


def _check_models(report: DoctorReport) -> None:
    from comic_translator.utils.di import _get_config_value, _load_config

    cfg = _load_config()
    bubble_model = str(_get_config_value(cfg, "bubble.model", "models/bubble/yolo11n-seg-mixed-bubble.pt"))
    p = Path(bubble_model)
    if p.exists():
        report.checks.append(CheckResult(OK, "模型", f"气泡检测 {bubble_model}"))
    else:
        report.checks.append(CheckResult(
            WARN, "模型", f"气泡检测权重缺失: {bubble_model}",
            hint="到模型库页/训练流程生成，或修改 config.yaml bubble.model",
        ))

    lama_dir = Path(_get_config_value(cfg, "render.inpaint.lama_model_path", "models/lama/big-lama"))
    has_lama = lama_dir.exists() and any(lama_dir.iterdir()) if lama_dir.exists() else False
    if has_lama:
        report.checks.append(CheckResult(OK, "模型", f"LaMa 权重 {lama_dir}"))
    else:
        report.checks.append(CheckResult(
            WARN, "模型", f"LaMa 本地权重缺失: {lama_dir}",
            hint="首次使用 LaMa 修复时会自动下载，或配置 opencv 修复引擎",
        ))


def _check_config(report: DoctorReport) -> None:
    from comic_translator.utils.di import _get_config_value, _load_config

    cfg_path = Path("config.yaml")
    if not cfg_path.exists():
        report.checks.append(CheckResult(
            WARN, "配置", "config.yaml 不存在",
            hint="复制 config.example.yaml 为 config.yaml",
        ))
        return
    try:
        cfg = _load_config()
        report.checks.append(CheckResult(OK, "配置", "config.yaml 解析正常"))
    except Exception as e:
        report.checks.append(CheckResult(FAIL, "配置", f"config.yaml 解析失败: {e}"))
        return

    mode = str(_get_config_value(cfg, "llm.mode", "ollama"))
    if mode == "api":
        import os
        import re
        raw = str(_get_config_value(cfg, "llm.api.api_key", ""))
        m = re.match(r"^\$\{(\w+)\}$", raw.strip())
        resolved = os.environ.get(m.group(1), "") if m else raw
        if resolved:
            report.checks.append(CheckResult(OK, "配置", "LLM API key 已配置（mode=api, 不回显）"))
        else:
            report.checks.append(CheckResult(
                FAIL, "配置", "LLM API key 未配置",
                hint=f"设置环境变量 {m.group(1) if m else '或在 config.yaml 直接填写'}",
            ))
    else:
        report.checks.append(CheckResult(OK, "配置", "LLM mode=ollama（本地）"))


def _check_data_dirs(report: DoctorReport) -> None:
    db = Path("data/comic_translator.db")
    report.checks.append(CheckResult(
        OK if db.exists() else WARN, "数据",
        f"项目数据库 {'存在' if db.exists() else '尚未创建（首次运行自动生成）'}",
        detail=str(db),
    ))
    for d in ("data/comics", "data/output"):
        p = Path(d)
        if p.exists():
            report.checks.append(CheckResult(OK, "数据", f"目录 {d}"))
        else:
            report.checks.append(CheckResult(WARN, "数据", f"目录缺失: {d}", hint="会按需自动创建"))


def run_doctor() -> DoctorReport:
    """执行全部体检项"""
    report = DoctorReport()
    _check_python(report)

    # 核心依赖（安装即有）
    for mod in ("yaml", "typer", "loguru", "sqlalchemy", "PIL", "numpy", "cv2", "pydantic"):
        _check_import(report, "核心依赖", mod, pip_name={"PIL": "pillow", "cv2": "opencv-python"}.get(mod, mod))

    # 可选扩展
    _check_import(report, "可选扩展", "gradio", extra="gui", required=False)
    _check_import(report, "可选扩展", "easyocr", extra="ocr", required=False)
    _check_import(report, "可选扩展", "torch", extra="gpu", required=False)
    _check_import(report, "可选扩展", "ultralytics", extra="yolo", required=False)
    _check_import(report, "可选扩展", "paddleocr", extra="paddle", required=False)
    _check_import(report, "可选扩展", "simple_lama_inpainting", extra="inpaint", required=False)

    _check_models(report)
    _check_config(report)
    _check_data_dirs(report)
    return report
