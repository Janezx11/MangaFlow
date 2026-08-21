# Changelog

本项目所有值得记录的变更。格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 计划中
- 微型 demo 数据集 + 导入脚本（让 ML 标注/训练对复刻者可体验）
- `download-models` 命令（从 HuggingFace 自动拉取模型）
- GitHub Actions CI（ruff + mypy + pytest）
- README 翻译前后对比图 / GUI 截图 / 项目 Banner
- 多语言 OCR（日语/中文）与 CBZ 导出（见 `PLAN.md`）

## [0.1.0] - 2026-08

首个版本。

### 新增
- **翻译管线**：文字区域检测（CTBD）→ OCR（EasyOCR）→ LLM 翻译 → 背景修复 → 译文回填渲染
- **气泡检测**：YOLO11-seg 实例分割，支持与文字区域融合
- **Web GUI（MangaFlow）**：翻译、项目数据、历史、设置四大主 Tab + 高级工具区
- **CLI**：`init` / `run` / `resume` / `list-projects`
- **翻译记忆**：精确匹配 + LIKE 模糊匹配 + 页面级缓存（重跑几乎免费）
- **角色一致性 & 术语表**：跨页记住角色称呼/风格；自定义术语词典 + 自动提取专有名词
- **批处理**：整本漫画批量处理，断点续跑
- **设置页**：GUI 内直接填写 API Key（掩码显示、保存即生效）、一键拉取可用模型
- **跨气泡纠错**："跨气泡手术"，自动识别被 OCR 合并的相邻气泡并拆分

### 修复
- 翻译失败的气泡不再把原文当译文渲染，按配置显示 `TRANSLATION FAILED` 标记或回退原文
- 渲染输出继承原始文件名（`{原文件名}_translated.png`）
- 模型库「启用」模型时同步 `config.yaml` 的 `bubble.model` 并热重建检测 provider
- 训练导出模型路径嵌套错误（`models/bubble/models/bubble/...`）
- GUI 数据集目录与训练脚本统一为 `dataset/`，修复 ML 训练/标注无数据可用
- `.gitignore` 的 `models/` 规则误伤源码包（已锚定 `/models/`）

### 性能
- 翻译文本去重：相同原文只调用一次 LLM，结果映射回所有同文气泡
- 并发翻译 worker（数量可配置）+ 令牌桶限流（RPS 可配置）

### 依赖与打包
- 补全 `pyproject.toml` 依赖声明：core 15 个 + 10 个 optional extras（gui/ocr/paddle/gpu/yolo/inpaint/vision/test/all 等）
- 新增 MIT License、`.gitattributes`（统一 LF 行尾）

### 文档
- README（简介/快速开始/安装/配置/模型/已知限制）
- `docs/`：架构、气泡、OCR 兼容性、批处理管线设计
- `OPTIMIZATION.md`、`RELEASE_CHECKLIST.md`、`PLAN.md`

[0.1.0]: https://github.com/Janezx11/comic-translator/releases/tag/v0.1.0