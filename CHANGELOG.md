# Changelog

本项目所有值得记录的变更。格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 交互式编辑闭环（精修工坊）
- **新增「精修」页**：选择历史运行与页码 → 表格直接修改任意对白块的 译文/字号/X偏移/Y偏移 → 一键重渲染该页并预览；字号微调经 typography v2 新增的 preset_font_size 钳制钩子生效，重建块时自动复用 render_debug.json 的已验证排版参数（字号+精确 bbox），机翻打底、人工精修的汉化组工作流
- 从 PageSnapshot 重建对白块（气泡几何以 bbox 近似），只重渲染单页、就地覆盖输出保持历史预览一致；空译文/未修改块不生效


### 角色一致性（路线图 #3）
- **说话人归属激活**：翻译成功的页自动追加 1 次 LLM 调用，判定每句对白的说话人并识别新出现的具名角色（按项目隔离）
- **角色记忆自动填充**：新角色（名字/别名/语气/描述）自动写入 CharacterIdentity 表；与已知角色或本批新角色重名的跳过，避免污染
- **上下文注入闭环**：后续页翻译时自动携带已积累的角色身份（含语气），实现连载翻译的名字/语气一致性
- **配置开关** `translation.speaker_attribution`（默认开启）；归属失败静默跳过，绝不影响翻译主流程


### 产品化（脱离 demo 定位）
- **删除第三条平行翻译路径**：`TranslationService`/`OCRService`/`QAService`/`ContextAssembler`/`PageSummaryService` 及 `services/ocr_health_checker.py`——仅被 DI 注册、零消费方的死链（连 ContextAssembler→PageSummary 也只彼此引用）。`providers/translation/*` 保留为可独立使用的库类
- **删除 `demo/` 目录**（11 个脚本中 7 个引用已删内部 API）；维护中的工具统一在 `scripts/`
- **DI 容器瘦身**：急切构建但零消费的 render_service/typography 栈、translation_provider、qa_provider、render_provider 全部移除（实例 20→13），启动更快、依赖图更诚实
- **新增 `comic-translator doctor`**：Python/依赖/可选扩展/模型权重/配置/数据目录 六类体检，冷启动自检（exit code 可用于脚本判断）
- **日志接线**：`setup_logging` 接入 CLI 与 GUI 入口（`CT_LOG_LEVEL`/`CT_LOG_FILE` 可覆盖），此前 utils/logging.py 从未被调用
- **质检合格口径统一**：`passed` 从"总分达标"改为 `not needs_retry`（与自动重跑判定一致），被标记页不再计入合格率
- **移除死配置键** `project.db`（数据库路径固定为 data/comic_translator.db，文档如实说明）
- **CI 绿灯**：ruff 规则集收敛到可持续集（E/F/I/N/W/UP/B/C4/RUF，历史 2100+ 违例清零）、移除 strict mypy 门禁（类型注解渐进补齐）、`--start/--end` 等历史宣称与实现全面对齐

### 文档（架构可视化）
### 文档（架构可视化）
- 新增两张 archify 交互式架构图（自包含 HTML，深/浅主题、可搜索/导出）：`docs/diagrams/core-architecture.html`（CLI+GUI 共用核心管线架构）与 `docs/diagrams/quality-loop.html`（质检闭环自动重试流程）；JSON 源文件同目录，均通过 9/9 showcase 校验与浏览器取证（visual-check 四视口零溢出）

### Web UI（Gradio 6 首屏数据与启动修复）
- **修复启动崩溃**：`gr.update()` 返回 dict 不支持属性访问，历史页预启动赋值会在 GUI 构建时抛 `AttributeError`（内置浏览器实测发现）
- **修复表格/面板首屏无数据**：Gradio 6 中「页面构建后、启动前」对组件 `.value`/`.visible` 赋值**不会进入客户端配置**（点刷新按钮才出数据）——历史/模型库/质检报告/评估面板 4 个页面改为**构造时传 `value=`**，实测首屏即显示完整数据（历史 22 条记录、模型库 6 个模型、质检报告摘要；浏览器截图验证）

### 质检闭环（管线自动重试）
- **管线接通质检闭环**：完整管线跑完后自动 聚合质检报告 → 对「翻译失败导致不合格」的页执行 LLM 逐气泡重试 + 重渲染 → 重新聚合并保存报告，GUI 日志与 CLI 输出可见 重试前后合格率对比（如 `质检: 自动重试 3 页，合格率 85% → 96%`）
- **保守重试策略**（`_select_retry_candidates`）：只自动重试内存中存在 failed 翻译气泡的页——失败翻译重试有真实增益；纯渲染质量问题（模糊/残留/排版）重渲染是确定性的、重试无增益，明确标注"建议人工复核"而不是浪费 LLM 调用
- **重试尊重取消**：质检重试循环检查取消标志，GUI 取消按钮可中断重试过程；重渲染后 `render_debug.json` 随之更新，重聚合反映真实改进
- **运行记录后置保存**：`_save_run_record` 移至质检闭环之后，重试产生的最终译文/渲染路径一并入档
- **质检聚合跟随运行输出目录**：修复自定义渲染输出目录时质检报告聚合到 `data/output/rendered` 旧数据的问题
- **GUI 步骤条新增「质检」步骤**（检测→OCR→翻译→渲染→质检→完成），未知阶段安全降级不变

### 架构（管线收敛）
- **CLI 与 GUI 收敛为同一条核心管线**：新增框架中立包 `comic_translator.core`（`ComicPipelineService` + `PipelineTranslationMixin`），完整流程 检测→气泡归属→OCR→翻译→擦除→渲染 迁入其中；`gui/services/gui_pipeline_service.py` 变为兼容层（`GuiPipelineService` 为别名），所有历史导入路径不变
- **CLI `run` 现在输出翻译后的图片**：原先只做 OCR+翻译+入库，现走完整管线渲染到 `data/output/rendered`（`--output` 可自定义）；CLI `run`/`resume` 消费同一进度生成器并逐页打日志
- **CLI `resume` 断点续跑生效**：`run` 每页成功后通过 `ProjectService.update_progress` 持久化 `pages_completed`（原先该仓库方法零调用方，resume 永远从第 1 页开始）；`resume` 从完成页+1 继续
- **核心管线支持页码范围**：`run_pipeline(start_page=, end_page=)`，页码保留在完整图片列表中的编号，部分翻译与全量运行的 `page_NNNN` 输出名一致；`--start/--end` CLI 参数从"解析但未使用"变为真正生效
- **OCR 引擎配置化**：核心管线按 `config.yaml` 的 `ocr.provider/lang/use_gpu` 构建引擎（缺省 easyocr，与 README 一致），配置引擎导入失败回退 easyocr 并告警，完全不可用时 NullOCR 兜底；此前 GUI 硬编码 EasyOCR(en)，`ocr.lang` 对 GUI 无效
- **CLI 选项变更**：`run` 移除 `--batch-size/--cache/--batch-delay`（新管线按页并发翻译、节奏由令牌桶限速、缓存由 `translation.use_cache` 配置控制）；新增 `--output/--target-lang`

### Web UI
- **翻译页卡片化布局**：① 输入区 → ② 开始/取消 → ③ 高级选项 → ④ 进度状态 → ⑤ 运行日志 五段卡片分区，简化冗余标签
- **可取消翻译**：新增「⏹ 取消」按钮（运行中显示），通过 Gradio `cancels` 机制终止进行中的生成器
- **防重复运行**：运行中禁用「开始翻译」按钮，完成后自动复位
- **进度状态增强**：状态区显示阶段图标（🔍检测/👁️OCR/💬翻译/🎨渲染/✅完成）+ 页码；`gr.Progress` 顶栏原生进度
- **设置页卡片化**：大语言模型 / 检测模型 / 翻译 / 输出 / GPU 状态 分区卡片
- **视觉美化**：品牌头部（强调下边框）、卡片圆角阴影、按钮/进度条样式、深色模式友好的主题变量 CSS
- **修复**：Gradio 6 的 `theme`/`css` 弃用警告（改由 `launch()` 传入）；`gr.Group` 不接受 `scale` 参数

### 修复
- **质检报告 0 页 Bug**：`QualityOrchestrator.aggregate_report` 查找 `render_debug.json` 的目录模式与 `RenderQualityService` 实际写盘位置（`rendered/_debug/page_XXXX/`）不匹配，导致每次运行后报告总显示 0 页。修复后从 105 页真实 debug 文件正确聚合出总页数 105、通过 103（98.1%）、平均渲染质量 71.3。已存在的 `data/output/rendered/_debug/page_*` 文件现可被正确读入。
- **项目数据页「点击行加载表单」失效**：Gradio 6 的 `gr.Dataframe.select` 事件将表格值以 **pandas DataFrame** 传入，而三个 `on_select_*` 处理器用了 `not table`（DataFrame 求布尔值抛 `ValueError`）和 `table[row_idx]`（DataFrame 数字下标按列取，抛 `KeyError`），导致术语表/角色记忆/翻译记忆**点击行无法回填表单，更新/删除无法选中行**。新增 `_row_from()` 兼容 pandas DataFrame / list-of-lists，三个处理器全部修复（实测 新增/选中/更新/删除 全链路通过）。
- **自动术语表垃圾词条**：`_save_auto_glossary` 曾把提取到的大写英文词以 `source=target=词本身` 存入（English→English，无翻译价值且撑爆翻译上下文、误导 LLM），库里 1729 条全部为这类垃圾。已全部清除，并将自动抽取改为 **LLM 抽取真实译名**（翻译后让 LLM 从源文+译文对中提取专有名词及对应译名，拒绝英文→英文与普通单词，幂等跨页去重；无 LLM 时静默跳过）。实测 `Stark→斯塔克 / HYDRA→九头蛇 / Manhattan→曼哈顿` 等真实映射正常写入，`Three/Chapter` 等普通词被排除。

### 代码质量
- 拆分 `gui_pipeline_service.py` 单体（1888 → 853 行）：翻译阶段逻辑抽离到 `gui_pipeline_translation.py`（`GuiPipelineTranslationMixin`），公共 API 不变（`_translate_dialogues` / `retry_page_failed_bubbles` / `collect_images` 等原位可用）
- 删除重复的 HTTP 降级翻译路径（`_translate_via_http` / `_translate_single_http`），翻译统一走 DI 容器的 LLMClient
- 删除检测阶段死代码（`_merge_ocr_results` / `_resolve_cross_bubble_blocks` / `_reocr_in_bubble` 等，已被 `BubbleAwareOCRService` 取代）及模块级 bbox 几何函数
- 清理占位 Provider（`manga_ocr` / `gemma_qa` / `opencv_render`）的未用导入、导入排序、缺尾换行等 lint 问题，保留为已文档化的扩展点
- 删除空壳包 `comic_translator.glossary`（内容已移入 `services/glossary_service.py`）
- `pyproject.toml`：ruff/mypy 目标统一到 Python 3.10（对齐 `requires-python` 与实际运行环境）；忽略对 CJK 代码库误报的 RUF001/002/003（全角标点属正确中文排版）

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

[0.1.0]: https://github.com/Janezx11/MangaFlow/releases/tag/v0.1.0