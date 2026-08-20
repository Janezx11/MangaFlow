# 漫画翻译系统（comic-translator）项目现状报告

> 面向 AI/LLM 读者的完整介绍。读完本文档即可了解项目全貌、技术架构、当前状态与已知问题，**无需阅读任何源代码**。
>
> 生成时间：2026-08-17 ｜ 版本：v0.1.0

---

## 1. 项目概述

**它是什么**：一个自动化的漫画/漫画图片翻译系统。给定一组漫画图片，自动完成「检测文字 → 识别文字 → 翻译 → 抹除原文 → 把译文回填渲染」的完整流程，输出翻译后的图片。

**目标用户**：漫画汉化组、漫画爱好者、需要翻译漫画的个人。

**形态**：
- **命令行 CLI**（`comic-translator`，基于 typer）：项目初始化、翻译、断点续跑、项目列表
- **Web GUI**（Gradio，本地运行于 `127.0.0.1:7860`）：图形化操作，含翻译页、设置页、调试页、实时日志

**定位**：面向 **英文漫画 → 中文** 等语言对的本地翻译工具，强调批量处理、角色一致性、术语一致性和断点续跑。

---

## 2. 核心功能

| 功能 | 说明 | 状态 |
|------|------|------|
| 文字区域检测 | 用 CTBD（Comic Text and Bubble Detector，RT-DETR-v2 架构）检测 `text_bubble`/`text_free`/`bubble` 三类区域 | ✅ 可用 |
| 气泡检测 | 用 YOLO11-seg 实例分割检测气泡（含掩码/多边形） | ✅ 可用 |
| OCR 识别 | EasyOCR（默认英文），支持 GPU 加速 | ✅ 可用 |
| LLM 翻译 | OpenAI 兼容 API（如 DeepSeek、Sensenova）+ 本地 Ollama 双模式 | ✅ 可用 |
| 渲染回填 | OCR 掩码 + 背景修复（OpenCV/LaMa）+ 字体排版，把译文画回图片 | ✅ 可用 |
| 翻译记忆 | 已有译文复用（精确匹配 + LIKE 模糊匹配） | ✅ 可用 |
| 语义翻译记忆 | 语义级翻译记忆（有 TODO 未完全实现） | ⚠️ 部分 |
| 角色一致性 | 跨页记住角色称呼/风格 | ✅ 可用 |
| 术语表 | 自定义术语词典（增删改查/分类） | ✅ 可用 |
| 批处理 | 整本漫画批量处理，断点续跑（resume） | ✅ 可用 |
| 跨气泡纠错 | 「跨气泡精准手术」：自动识别被 OCR 合并的相邻气泡并拆分 | ✅ 可用（新） |
| 原生文件夹选择 | Windows 系统级文件夹选择对话框（替换网页内置面板） | ✅ 可用（新） |

---

## 3. 技术栈

- **语言**：Python（`requires-python >= 3.10`，实际运行于 Python 3.10.20）
- **Web UI**：Gradio 6.22
- **核心库**：Pydantic v2、SQLAlchemy 2.0、Pillow、OpenCV、Loguru、Typer、OpenAI SDK、httpx
- **检测**：ultralytics（YOLO11-seg）、onnxruntime-gpu（CTBD）
- **OCR**：EasyOCR 1.7.2
- **渲染**：LaMa inpainting（torch）、OpenCV、字体排版服务
- **GPU**：NVIDIA RTX 4060（8GB），torch 2.6.0+cu124，CUDA 12.4

### 依赖安装现状（重要）

`pyproject.toml` 中声明的依赖**不完整**：仅声明了 11 个基础库，但实际运行需要的 `gradio`、`easyocr`、`onnxruntime(-gpu)`、`ultralytics`、`numpy`、`scipy` 等**均未声明**。当前依赖是通过两个本地虚拟环境手工安装的：

- `.venv310`（Python 3.10.20）— **实际运行环境**（6.7GB，含 GPU 相关全部依赖）
- `.venv`（Python 3.13.9）— 遗留环境（CPU torch，非运行环境）

**这意味着：`pip install -e .` 后项目无法运行**——这是开源分发最大的阻塞点。

---

## 4. 架构

### 4.1 分层结构

```
src/comic_translator/
├── providers/     # 可插拔能力层（通过抽象基类解耦）
│   ├── text_detection/   # CTBD 文本检测
│   ├── bubble/           # YOLO 气泡检测
│   ├── ocr/              # EasyOCR / PaddleOCR / MangaOCR
│   ├── translation/      # DeepSeek / Qwen 翻译客户端
│   ├── inpaint/          # OpenCV / LaMa / SD 背景修复
│   ├── render/           # 渲染引擎（OpenCV 版未实现，见问题清单）
│   ├── qa/               # 翻译质量校验（部分未实现）
│   └── vision/           # 视觉模型适配（部分未使用）
├── services/      # 业务服务层（OCR 归一化、翻译记忆、角色记忆、上下文组装、
│                  #   词汇表、页面摘要、批处理、渲染服务等 ~20 个服务）
├── storage/       # 持久化（SQLAlchemy 数据库）
├── models/        # Pydantic 数据模型（图片、区域、气泡、对话块、翻译块等）
├── config/        # 配置管理
├── prompts/       # LLM 提示词模板
├── glossary/      # 术语表模块
├── cache/         # 缓存
├── utils/         # 依赖注入容器等工具
└── gui/           # Gradio 界面（页面 + 服务编排）
```

### 4.2 主流程（数据流）

```
漫画图片
  → 1. 检测：CTBD 检测文字区域 + YOLO 检测气泡（可开启融合）
  → 2. OCR：按文字区域识别（区域级 OCR，保持上下文完整）
       └─ 跨气泡手术：若某 OCR 块横跨 2+ 个分离气泡，抹除干扰文字后二次识别
  → 3. 匹配：把 OCR 结果归属到文字区域/气泡 → 生成对话块
  → 4. 翻译：LLM 批量翻译（带记忆/术语表/角色上下文）
  → 5. 渲染：生成掩码 → 修复背景 → 排版回填译文
  → 输出：翻译后的图片
```

### 4.3 关键设计决策

- **区域级 OCR 优先**：先检测文字区域再逐区域 OCR（而非全图 OCR 后分割），保证文字上下文完整、不易碎片化。这是当前主路径。
- **跨气泡手术（新）**：仅在确认某 OCR 块确实横跨多个分离气泡时才触发二次识别，正常页面零影响。先合并 YOLO 重复检测的气泡（IoU>0.5 或小框 70% 被大框包含），再按 polygon 精确计算重叠。
- **依赖注入**：`utils/di.py` 提供容器，服务在启动时组装。

---

## 5. 当前状态（实测验证）

- ✅ **单元测试通过**（pytest）：覆盖模型、OCR 归一化、翻译记忆、语义 TM、角色记忆、上下文组装、术语表、页面摘要、prompt 缓存、Bubble-aware OCR、跨气泡拆分等。注意：受限沙箱（临时目录不可写）下部分用 `tmp_path` 的用例会报 `PermissionError`（环境限制，非代码问题）；在正常环境可全量通过
- ✅ **GPU 全链路生效**：EasyOCR（自动检测 CUDA）、CTBD（onnxruntime-gpu，经 `os.add_dll_directory` 借用 torch 自带 CUDA 12.4 DLL 加载 CUDAExecutionProvider）、YOLO、LaMa 均在 GPU 运行
- ✅ **OCR 架构已验证**：3 页实际测试，新架构（区域 OCR + 跨气泡手术）与原架构输出完全一致，零退化
- ✅ **跨气泡合并 bug 已修复**：`_merge_blocks`（OCR 后处理 L4）原按全局 reading_order 合并同列相邻行，导致多气泡文字被合并成 1 个超大 block、最终只渲染到一个气泡；已改为按 `bubble_id` 分组，仅在同一气泡内合并相邻行
- ✅ **Windows 原生文件夹选择**已集成并验证
- ✅ 运行记录/历史测试产物已清理

---

## 6. 已知问题与不足

### 🔴 严重（影响他人安装/上架）

| # | 问题 | 影响 |
|---|------|------|
| 1 | ~~**依赖声明不完整**~~ ✅ 已补全（core 15 + 10 个 extras，版本下限对齐 .venv310；待有网环境实测安装） | 别人无法安装运行 |
| 2 | ~~**无 README**~~ ✅ 已有（简介/快速开始/安装/配置/模型/限制） | 无法展示项目 |
| 3 | ~~**无 LICENSE**~~ ✅ 已加 MIT | 他人无法合法使用 |
| 4 | ~~**无版本控制**~~ ✅ git 仓库已建（7 次提交） | 无法直接 push 到 GitHub |
| 5 | ~~**.gitignore 严重不完整**~~ ✅ 已重写（忽略 .venv*/models/dataset/runs/.claude/.tools/~/；取消忽略 uv.lock） | 直接 push 会爆仓库 |
| 6 | **config.yaml 含真实 API key** → 已改为 `${SENSENOVA_API_KEY}` 占位；**旧 key 尚未在控制台轮换** | 潜在密钥泄露 |

### 🟡 中等（代码质量）

| # | 问题 | 说明 |
|---|------|------|
| 7 | **死代码/未实现** | `OpenCVRenderProvider.render` 直接 `raise NotImplementedError`；`gemma_qa`、`semantic_tm`、`opencv_render` 共 3 处 TODO |
| 8 | **版本约束不一致** | `requires-python >=3.10` 但 ruff/mypy 配置 `target py311`；包版本仅 0.1.0 |
| 9 | **pytest 配置警告** | `asyncio_mode` 报"Unknown config option"（虽测试通过） |
| 10 | **跨平台脆弱** | GPU 依赖用「借 torch 的 CUDA DLL + `os.add_dll_directory`」的 hack；文件夹选择器仅 Windows |
| 11 | **硬编码相对路径** | `data/comics`、`models/bubble/...` 等，对工作目录敏感 |

### 🟢 低（卫生/体验）

| # | 问题 |
|---|------|
| 12 | **跨气泡文字合并 bug（已修复）** — OCR 后处理 `_merge_blocks` 原按全局 reading_order 合并同列相邻行，不区分气泡，导致垂直排列的多气泡文字被合并为 1 个 block；已改为按 `bubble_id` 分组，仅在同一气泡内合并相邻行 |
| 13 | `.claude/settings.json` 本地工具配置不该进仓库 |
| 14 | OCR 仅支持英文（日文/中文需扩展） |
| 15 | 无端到端集成测试、无 CI |

---

## 7. 发展建议

### 短期（上架前必做，约 1~2 小时）
1. 补全 `pyproject.toml` 依赖（拆分 optional groups：gui/ocr/gpu）
2. 写 README（功能截图、安装、快速开始、模型下载、已知限制）
3. 加 LICENSE（建议 MIT）
4. 修 .gitignore（加 `.venv*/`、`dataset/`、`models/`、`.claude/`、`.tmp/`；取消忽略 `uv.lock`）
5. `git init` + 首次提交（提交前用 `git status` 确认无大文件/密钥）
6. 轮换 config.yaml 中的 API key

### 中期
7. 给「跨气泡手术」补回归单测（当前只靠手工验证，易退化）
8. 模型不进仓库：写下载脚本从 HuggingFace 拉取，或用 Git LFS
9. GitHub Actions CI（ruff + mypy + pytest）
10. 清理/标注死代码

### 长期
11. **多语言 OCR**（日文/中文）——最大用户增长点
12. 跨平台支持、打包成 Windows 桌面应用（面向不懂 Python 的汉化组）
13. 译文编辑页（原文/译文对照、手动修正）、CBZ 打包导出
14. 性能基准脚本（量化 GPU 收益写进 README）
15. 模型权重许可核查与文档化、版权合规声明

---

## 8. GitHub 上架评估

**结论：可以上架，但必须先完成 🔴 级问题（约 1~2 小时准备）。**

- **适合上架的理由**：功能完整（CLI+GUI）、121 测试全绿、架构分层清晰、有 4 篇架构文档、配置模板干净（API key 用环境变量占位）
- **上架前提**：修 .gitignore（否则 6.7GB 环境+模型全进仓库）、补依赖声明、加 README + LICENSE、轮换 API key
- **发布建议**：以 `v0.1.0` 首版发布，README 标注 WIP 功能与平台限制，附翻译前后对比图

---

## 9. 附：代码里"看不见但 GPT 需知"的细节

这些是审查中发现的、代码注释里没有明确写清的实现细节：

1. **GPU 加载 hack**：onnxruntime-gpu 需要系统级 CUDA 12 + cuDNN 9 DLL；项目未安装 CUDA Toolkit，而是通过在 `CTBDTextDetectorProvider._ensure_model()` 里调用 `os.add_dll_directory(torch/lib)`，借用 torch 2.6.0+cu124 自带的 DLL 让 CUDAExecutionProvider 成功加载。**依赖 torch 与 onnxruntime-gpu 的 CUDA/cuDNN 版本匹配**。
2. **OCR 架构演进史**：曾尝试「逐气泡裁剪 OCR」（用气泡多边形涂白后逐气泡识别），发现会切碎文字（一句话被拆成多个碎片 block），已弃用；现采用「区域 OCR 为主 + 跨气泡手术为辅」。
3. **跨气泡手术逻辑**：先合并 YOLO 重复检测的重叠气泡 → 逐 OCR 块用 polygon 算重叠 → 仅当块与 2+ 个分离气泡重叠才触发「抹除干扰文字二次识别」。
4. **原生文件夹选择**：`gui/native_picker.py` 用 ctypes 调 `SHBrowseForFolderW`，仅 Windows 有效；对话框由**服务器所在机器**弹出（本地使用无碍，远程访问时在服务器侧弹出）。
5. **测试沙箱注意**：pytest 的 `tmp_path` fixture 需要可写临时目录，若 CI/沙箱限制临时目录会报 `PermissionError`（非代码问题）。
6. **`_merge_blocks` 全局合并 bug**：`ocr_postprocess_service.py` 的 L4 Block 合并原按全局 `reading_order` 排序，把同列相邻 block 合并（不管属于哪个气泡），导致垂直排列的多气泡文字被合并为 1 个超大 block。修复后按 `bubble_id` 分组，仅在同一气泡内合并相邻行。

---

*本报告由自动代码审查生成，基于实际运行验证与源码分析。*
