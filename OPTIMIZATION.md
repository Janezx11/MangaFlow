# 项目优化核查报告（OPTIMIZATION）

> 🛠️ **维护者文档**：项目内部优化清单与历史核查记录，公开读者可跳过。
>
> 生成时间：2026-08-17 ｜ 依据：对 `G:\trainslate` 的实际核查（源码、配置、git、测试、目录占用）
>
> 本报告只做**核查与建议**，未改动任何代码/文件。所有结论均经实测验证，非文档转述。
> 与已有 `PLAN.md`、`PROJECT_STATUS.md` 的差异点已单列（见 §4），建议以此报告为准更新。

---

## 0. 一句话结论

之前文档（`PROJECT_STATUS.md` §7）把"上架前必做"列为约 1~2 小时的工作，**但经核查这组工作目前一项都还没做**；此外还发现文档状态已过期（代码已实现但 PLAN 仍标"待开始"）、一个 1801 行单体文件、若干残留死代码、以及测试在本环境无法全绿（沙箱限制）。

---

## 1. 🔴 P0 — 上架/分发阻塞（全部未解决）

| # | 问题 | 实测证据 | 建议改法 | 工作量 |
|---|------|---------|---------|--------|
| 1 | **无 git 仓库** | `git status` → `fatal: not a git repository` | `git init` + 首次提交（先修 #3 再提交） | 5 分钟 |
| 2 | **无 README / 无 LICENSE** | 根目录均不存在 | README：功能/安装/快速开始/模型下载/已知限制；LICENSE 建议 MIT | 1 小时 |
| 3 | **.gitignore 错误** | 误忽略 `uv.lock`（第 29 行）；未忽略 `.venv310`(6.9GB)、`.venv`(1.4GB)、`dataset`(163MB)、`models`(115MB)、`runs`(22MB)、`.claude/`、`~`(79MB) | 加 `.venv*/`、`dataset/`、`models/`、`runs/`、`.claude/`、`~`；**取消忽略 `uv.lock`** | 10 分钟 |
| 4 | **依赖声明不完整** | `pyproject.toml` 仅 11 个基础库；`gradio`、`easyocr`、`onnxruntime-gpu`、`ultralytics`、`numpy`、`scipy` 等未声明 → `pip install -e .` 无法运行 | 补全依赖并拆 optional groups（gui/ocr/gpu/yolo） | 30 分钟 |
| 5 | **config.yaml 含真实 API key** | 已从 config.yaml 移除，改用 `${SENSENOVA_API_KEY}` 环境变量占位；需在 Sensenova 控制台轮换旧 key | 换成 `${SENSENOVA_API_KEY}` 占位 + 文档说明；上架前轮换该 key | 10 分钟 |

**执行顺序**：修 #3 → #1 `git init` → #2 → #4 → #5。提交前用 `git status` 确认无大文件/密钥。

---

## 2. 🟡 P1 — 代码质量

| # | 问题 | 实测证据 | 建议 |
|---|------|---------|------|
| 6 | ~~**`gui_pipeline_service.py` 为 1801 行 / 76KB 单体**~~ ✅ 已拆分 | 翻译阶段抽离到 `gui_pipeline_translation.py`（`GuiPipelineTranslationMixin`），服务文件 1888 → 853 行；公共 API 不变 | 拆成子模块（ocr 编排 / 翻译 / 渲染 / 进度）；两个翻译路径共用一处上下文构造 |
| 7 | ~~**重复的 HTTP 降级翻译路径**~~ ✅ 已删除 | `_translate_via_http` / `_translate_single_http` 已移除，翻译统一走 DI 容器的 LLMClient | 删降级路径，统一走 DI 的 LLMClient；或抽公共 `_build_system_prompt()` |
| 8 | ~~**残留死代码（空壳）**~~ ✅ 已清理 | `glossary/manager.py` 已删、空包 `comic_translator.glossary` 已删；检测阶段 `_merge_ocr_results` / `_resolve_cross_bubble_blocks` / `_reocr_in_bubble` 等死代码已删（被 `BubbleAwareOCRService` 取代）；`manga_ocr` / `gemma_qa` / `opencv_render` 保留为文档化扩展点并清理 lint | 删除或标注 `# deprecated`，避免误导调用方 |
| 9 | ~~**`typography_service.py`(v1) 被 v2 取代仍导出**~~ ✅ 已解决 | `services/__init__.py` 已只导出 `TypographyServiceV2` | 移除 v1 导出，或标记废弃 |
| 10 | ~~**`defaults.yaml` 未删**~~ ✅ 已解决 | `src/comic_translator/config/defaults.yaml` 已不存在 | 删除该文件，配置统一走 `config.yaml` |
| 11 | ~~**版本/目标不一致**~~ ✅ 已统一 | `requires-python >=3.10` 且 ruff/mypy 现统一为 py310（对齐实际运行环境 3.10.20）；另在 ruff 中忽略对 CJK 误报的 RUF001/002/003 | 统一到 3.10 或提升至 3.11 并说明 |
| 12 | **测试文档状态过期** | 文档称"121~124 全绿"，本次实测 **80 passed + 54 errors**（54 个全为 `tmp_path` 的 `PermissionError`，系本沙箱环境限制 `C:\Users\...\Temp\...` 写入，非代码 bug） | 在文档中注明验证环境与沙箱限制；建议 CI 里跑全量 |

---

## 3. 🟢 P2 — 卫生 / 体验

- `data/output`(178MB)、`runs/`(22MB，ultralytics 训练产物)、`~`(79MB EasyOCR 模型缓存) 建议进 ignore 或清理。
- **两套虚拟环境共 8.3GB**：`.venv310`（真运行环境）与 `.venv`（遗留 CPU 环境）。建议删除 `.venv`，并在 README 写清只用 `.venv310`。
- **无 CI、无端到端集成测试**（PLAN §2.5 未做）。建议 GitHub Actions：ruff + mypy + pytest。
- **硬编码相对路径**（`data/comics`、`models/bubble/...`）对工作目录敏感 → 建议从 `config.yaml` 统一读取（PLAN §2.4）。
- **跨平台脆弱**：GPU 用"借 torch 的 CUDA DLL + `os.add_dll_directory`" hack；文件夹选择器仅 Windows（`native_picker.py`）——文档已记录，属已知取舍。
- **ML 模型开发功能对 reviewer 不可见**：`dataset/`（真实标注数据）被 gitignore，clone 后标注/训练显示无数据集 → 建议附带**微型 demo 数据集 + 导入脚本**（见 PLAN §3.9），让功能可实际点通。

---

## 4. 文档与代码不一致清单（建议更新 PLAN.md / PROJECT_STATUS.md）

| 文档说法 | 实际代码状态 |
|---------|-------------|
| PLAN §2.1「术语表 + 角色记忆接入 GUI 翻译」标 📋 待开始 | **已实现**：`_build_translation_context`（gui_pipeline_service.py:1112）已从 DI 取 glossary + character identity |
| PLAN §2.2「OCR 后处理接入 GUI」标 📋 待开始 | **已实现**：gui_pipeline_service.py:282/791 已实例化并调用 `OCRPostprocessService.process_page()` |
| PLAN §2.6「翻译提速」标 ✅ 已完成 | 属实：TM 精确匹配 + 页面缓存复用（`_translate_dialogues` 1289 起）、并发 worker、自然排序 |
| PROJECT_STATUS §5「121 个单元测试全部通过」 | 需注明环境；本沙箱下 75 过 54 错（全为 tmp_path 权限） |

> 结论：**2.1、2.2 可从"待开始"改为"已完成"**；PROJECT_STATUS 的测试数字补环境说明。

---

## 5. 建议执行顺序（后续可选）

1. **上架组（P0）**：修 .gitignore → git init → README + LICENSE → 补依赖 → 轮换 key。
2. **P1 快速清理**：删空壳/死代码、删 `defaults.yaml`、移除 typography v1 导出（一次性、低风险）。
3. **P1 重构**：拆分 `gui_pipeline_service.py` 单体 + 去重 HTTP 降级路径（中等风险，需回归测试覆盖）。
4. **文档对齐**：按 §4 更新 PLAN.md / PROJECT_STATUS.md。
5. **P2 工程化**：CI、端到端测试、配置统一化、清理 `.venv` 与训练产物。

---

*本报告为核查结果，未改动任何文件。*
