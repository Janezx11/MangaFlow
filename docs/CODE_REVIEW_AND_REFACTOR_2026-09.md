# 代码审查与架构收敛记录（2026-09-06）

> 🛠️ **维护者文档**：本次维护会话的完整改动记录——全库代码审查与 bug 修复、路线图 #1「管线收敛」、#2「质检闭环」的实施细节。公开读者可跳过，主要入口见 `README.md`。
>
> 生成时间：2026-09-06 ｜ 基线：commit `7c75648`（ui phase3b-stability-fullwidth）之上
>
> **所有修复均经以下验证**：全量测试套件（134 → 147 通过）、`compileall` 语法检查、改动文件 ruff F/E9 类检查、针对每个关键修复的行为验证脚本、两轮脱机端到端管线运行（产出真实渲染图）。
>
> 注：工作区中 `settings_page.py`、`manga_ocr_provider.py`、`gemma_qa_provider.py`、`opencv_render_provider.py`、`OPTIMIZATION.md`、`PROJECT_STATUS.md`、`docs/UI_LAYOUT_STABILITY_AUDIT.md` 的改动**早于本次会话**（前一轮 UI 稳定性工作），不在本文范围内。

---

## 0. 一句话总结

全库审查发现并修复 **25+ 个真实 bug**（含 3 个必然崩溃的 P0、一个静默污染译文的数据 bug、一个内存 4 倍浪费的性能问题）；随后按路线图完成两项架构演进：**CLI 与 GUI 收敛为同一条核心管线**（CLI 从"只翻译不渲染"变为输出完整翻译图片、resume 断点续跑真实生效），以及**质检闭环接入管线**（跑完自动重试翻译失败页，产出重试前后合格率对比）。

---

## 1. 全库代码审查：已修复的 Bug

### 1.1 P0 —— 必然崩溃

| # | 位置 | 问题 | 修复 |
|---|------|------|------|
| 1 | `services/bubble_matching_service.py` `_split_cross_bubble_blocks` | `result.append(left_block)` 引用未定义变量（实际叫 `block_a`/`block_b`），任何 OCR 块与 2+ 气泡重叠 >5% 即抛 `NameError`；日志里的 `split_x` 在水平分割分支也未定义 | 改用正确变量名；`split_x if dx >= dy else split_y` 按分支取值 |
| 2 | `gui/pages/debug_page.py` `_show_from_history` | "找不到图片"分支返回 9 个值，Gradio 输出组件有 10 个 → 该事件必报 `ValueError` | 补上缺失的第 10 个返回值 |
| 3 | `services/quality_orchestrator.py` `retry_page` | `pt` 只在提供渲染服务时赋值，纯翻译路径（第 8 步）引用未绑定变量 → `UnboundLocalError` | `pt` 提前赋值并以此判定第 7 步 |

### 1.2 P1 —— 严重逻辑错误

**翻译数据污染类（最影响产物质量）：**

- **`providers/translation/qwen_translator.py` / `deepseek_translator.py`**：LLM 失败时 `translated_text = source_text`，下游 `_prepare_render_text` 把"译文==原文"归为 `unchanged` 而非 `failed` → **未翻译的英文直接渲染进中文漫画**，且重试机制永不触发。改为失败留空，渲染层按 `fallback_to_source` 显式决策。缺译文块同理留空（原用英文原文冒充）。
- **`services/ocr_postprocess_service.py` OCR_FIXES**：`gave→give`、`alone→along`、`lone→long`、`woods→Woods`、`stark→Stark` 等**把正确 OCR 单词改错**。删除全部"合法词→另一个词"映射（只留真正的误读修复如 `wepe→were`），同时清理重复字典键（`eiFT`/`thereis`/`gpeece` 各重复两次）和不可达的重复 `return`。
- **`gui/services/.../gui_pipeline_translation.py`（现 `core/translation.py`）**：`lstrip("0123456789.- ")` 把以数字/横杠开头的正常译文剥坏（`3.14是圆周率`→`14是圆周率`、`——好极了`→`好极了`）。改为 `_NUMBERING_PREFIX_RE`（点号后负向前瞻不跟数字）只剥一次序号前缀。
- **`services/text_normalizer.py`**：重复字典键 `'"': '“', '"': '”'` 使**所有**直引号变闭引号（`"Hello"`→`”Hello”`）。改为正则按位置区分开/闭引号。
- **`services/batch_prompt_builder.py` `parse_response`**：`source = corrected or ""` ——注释承诺"回退原文"但未实现，模型漏 corrected 时译文块 source 存空 → 渲染层静默跳过。补上 `batch_input.pages` 的 id→原文映射回退。

**几何/图像类：**

- **`providers/bubble/yolo_bubble_provider.py`**：`result.masks.data` 仍在 letterbox 后的网络输入分辨率，代码直接 `cv2.resize` 拉到原图 → **非正方形漫画页 mask 与 bbox 整体错位**，气泡归属、排版全受影响。改用 ultralytics 已映射回原图空间的 `result.masks.xy` 重建 mask（保留 letterbox undo 的 fallback 路径）；同时 mask 由全页嵌套浮点列表（每气泡约 100MB 级 Python float 对象）改为 0/1 整数列表（缓存单例，内存约降 4 倍）。
- **`providers/inpaint/base_inpaint_provider.py` `_to_bgr`**：对 ndarray 输入无条件 `RGB2BGR` 再翻转一次 → OpenCV 修复输出红蓝颠倒，且与 LaMa provider（按 BGR 约定）互相矛盾。统一契约：ndarray=BGR（直通），PIL=RGB（转换）。

**并发/网络类：**

- **`clients/openai_compatible_client.py`**：`APIConnectionError` 是 `APIError` 子类且无 `status_code`，落进"非 5xx 不重试"分支 → **网络抖动一次即整页失败**，配置的 `max_retries` 形同虚设。增加专门的连接错误重试分支；空 `choices`（200 但内容被过滤）改为确定性错误直通不重试。
- **`clients/ollama_client.py`**：`temperature`/`max_tokens` 放在 payload 顶层被 Ollama 静默忽略 → 翻译实际跑在模型默认 ~0.7 温度上，JSON 输出不稳定；`max_tokens` 不生效可能截断译文。统一移入 `options`（`num_predict`）。
- **两个客户端 `generate_stream`**：中途断流后重试会**从头重发全部内容**，下游拼接出双倍乱码。已输出内容后（`yielded_any`）禁止重试、直接抛错。
- **`utils/di.py`**：环境变量未设置时 `api_key or raw_key` 回退到 `"${VAR}"` **字面量**当 Bearer key 发出去（只产生难排查的 401）。改为启动时告警 + 请求时明确报错（不在容器构造期抛错，避免炸掉 GUI 启动）。
- **`gui/services/gui_pipeline_service.py`（现 `core/pipeline.py`）取消路径**：Gradio `cancels` 只关闭生成器，后台检测/OCR/翻译线程继续空跑并报 "Event loop is closed"；且 producer 的哨兵 `put` 在取消路径会死锁在满队列上。重写 `_run_sync`（finally 中 `gen.aclose()` 注入 GeneratorExit 优雅关闭）、主循环包 try/finally 保证 task 清理与 `_events_iter.aclose()`、producer finally 检查取消标志跳过哨兵；新增 `cancel_run()` API 并接到 GUI 取消按钮。

**GUI 功能失效类：**

- **`gui/pages/history_page.py` / `model_page.py` `on_select`**：`table[row_idx]` 作用在 Gradio 6 的 pandas DataFrame 上按列名取列 → 每次点击 `KeyError` → **删除运行、启用/禁用/删除模型功能完全无法使用**。改用 `iloc`（与 `project_manage_page._row_from` 的既有正确范例一致）。
- **`gui/pages/debug_page.py` `_do_retry`**：找图硬编码 `data/comics`/`data/input`，用户自选目录的重试静默跳过重渲染；且未传 `output_dir`，重渲染落到错误目录。改为 `_find_image(file_name, extra_dir=get_run_input_dir(pipeline_service.current_run_id))`。
- **`gui/pages/debug_page.py`**：OCR 覆盖图重建失败的 `except: pass` 静默吞错，历史页看起来"没有 OCR 数据"。镜像检测分支的处理，把错误写入 meta 信息。
- **`gui/pages/history_page.py` / `quality_report_page.py`**：页面构建时调用 `refresh_table()`/`refresh(...)` 但丢弃返回值——"自动刷新"是 no-op，首屏永远显示占位符。改为预启动赋值（与 `benchmark_page` 一致）。
- **`gui/pages/project_manage_page.py`**：删除 `refresh_glossary`/`refresh_chars` 死函数（未绑定任何按钮，且每个会重复查询两次 DB）。
- **`gui/native_picker.py` `_make_callback`**：`ctypes.c_wchar_p(initial_dir)` 的临时对象在表达式结束后即被 GC，`BFFM_SETSELECTION` 收到的 lParam 指向**已释放内存**（use-after-free）。改为闭包持有的 `create_unicode_buffer`。

**服务层数据/隔离类：**

- **`services/semantic_tm_service.py`**：`TranslationService` 调用 `store(project_id=...)` 但签名没有该参数 → 每次翻译成功后 TM 写入全部抛 `TypeError`。补齐 `project_id` 参数并持久化；`find_similar` 同步支持项目过滤（此前跨项目翻译记忆互相污染）。
- **`models/batch_result.py`**：`quality_orchestrator` 传 `entity_context=...` 给 `BatchInput`，Pydantic 静默丢弃 → prompt builder 的 `getattr(..., "")` 永远为空。补上字段。

### 1.3 P2 —— 静默失效/小问题

| 位置 | 问题 | 修复 |
|------|------|------|
| `services/bubble_postprocess_service.py` | 顶层无条件 `from scipy.spatial import ConvexHull` 在 try/except 守卫**之前**，scipy 缺失时整个包无法导入，fallback 是死代码 | 删除无条件导入，缺失时 `ConvexHull = None` |
| `providers/ocr/paddle_ocr_provider.py`、`providers/text_detection/ctbd_text_detector.py`、`providers/vision/locate_anything_provider.py` | `_ensure_initialized()`（可能构建 PaddleOCR / HF 下载 / 加载 3B 模型）在事件循环上同步执行，首页卡死 UI 数秒到数分钟 | 移入 `asyncio.to_thread`（与 YOLO provider 一致） |
| `providers/inpaint/lama_inpaint_provider.py` `_load_model` | 配置 `models/lama/big-lama` 被当作 `model_dir` 后代码再拼一级 `big-lama` → 双重嵌套永不命中，本地模型被静默忽略并触发默认下载 | 候选目录依次尝试（`model_dir` 本身或其下 `big-lama/`），失败再回退默认 |
| `utils/rate_limiter.py` | `wait_time += wait_time` 是翻倍不是累积，返回的"已等待秒数"错误 | 独立 `waited_total` 累积 |
| `services/project_service.py` `get_page_paths` | 按扩展名逐个 glob 再拼接 → 所有 png 排在 jpg 前、组内 `10.jpg` < `2.jpg`，页码与阅读顺序不符 | 一次收集 + `natural_key` 自然排序 |
| `services/render_qa_service.py` | 读取不存在的 `block.layout` 字段 → Overflow 与 Tiny-font 两个 QA 检查**永不生效** | Overflow 用 `bubble.bbox`，Tiny-font 用 `font_size`（0 表示未排版则跳过） |
| `providers/translation/*_translator.py` `_fallback_parse` | JSON 解析失败时把 `{"id": "b1", ...},` 这类**语法残片行**当译文渲染 | 跳过 JSON 残片行、用正则提取 translated 值，提取不到留空 |
| `services/quality_orchestrator.py` `retry_page` | 找 `render_debug.json` 只查一种布局，与 `RenderQualityService` 实际写盘位置（`rendered/_debug/page_NNNN/`）不匹配 → 重试成功也报 `render_quality=0`，反复触发无谓 LLM 重试 | 兼容三种布局（与 `aggregate_report` 一致） |

### 1.4 "宣传与现实不符"（已修）

| 问题 | 修复 |
|------|------|
| CLI `--start/--end` 解析后**从未使用**（`BatchTranslationService.run` 无此参数） | 服务层加页码范围过滤并接线（后随管线收敛迁移到 `core/pipeline.select_page_range`） |
| `pyproject.toml` 允许 `paddleocr>=2.9.0`（含 3.x），但代码用 3.x 已删除的 `use_gpu`/`show_log` 参数，全新安装必崩 | 钉 `paddleocr>=2.9.0,<3.0` |
| `utils/di.py` 默认 PaddleOCR 分支不传 `lang`/`use_gpu` → `config.yaml` 的 `ocr.lang` 对 CLI 无效 | 传入配置值 |
| `config.example.yaml` `background_cleaning` 重复键（行 49 true 被行 65 false 覆盖，用户以为默认 true） | 删除重复键 |
| README 声称 15 个 GUI 页面，实际 14 | 修正 |

---

## 2. 路线图 #1：管线收敛（CLI 与 GUI 同一条核心管线）

**背景**：GUI 管线（`gui_pipeline_service`，完整流程）与 CLI 批量服务（`BatchTranslationService`，仅 OCR+翻译+入库）是两套平行实现，行为分叉导致本次审查中多处 bug 成对出现；README 宣传的"输出翻译后的图片"只有 GUI 能达成；`resume` 因 `pages_completed` 无任何写入调用方而永远从第 1 页开始。

### 2.1 新增 `comic_translator/core/` 包（框架中立）

- `core/pipeline.py`：完整流程编排（检测 CTBD+YOLO → 气泡归属 → OCR → 翻译 → 擦除 → 渲染），类名 `ComicPipelineService`。**不 import gradio、不依赖 gui 命名空间**。
- `core/translation.py`：翻译阶段逻辑（原 `GuiPipelineTranslationMixin` → `PipelineTranslationMixin`）。
- `gui/services/gui_pipeline_service.py` / `gui_pipeline_translation.py` 变为**兼容层**（约 20 行 re-export），`GuiPipelineService` 是别名——7 个 GUI 页面与 2 个测试文件的历史导入路径**零改动**（`assert Alias is ComicPipelineService` 验证同一性）。

### 2.2 CLI `run` / `resume` 重写（`__main__.py`）

- `run` 现在走完整管线渲染到 `data/output/rendered`（`--output` 自定义），逐页打 `[stage]` 进度日志，结束汇总"N 页成功 / M 页失败"。
- `resume` 真实生效：`run` 每页成功后调用 `ProjectService.update_progress` 持久化 `pages_completed`（此前该方法全仓库零调用方），`resume` 从完成页 +1 继续，含"已完成"判定（用实际图片数而非快照 `total_pages` 判定，容忍中途加页）。
- 页码范围：核心管线新增 `select_page_range()`——**页码保留在全量列表中的编号**，从第 3 页续跑输出仍是 `page_0003` 而非 `page_0001`，渲染文件名与全量运行一致。
- ⚠️ **破坏性变更**：`run` 移除 `--batch-size/--cache/--batch-delay`（新管线按页并发 + 令牌桶限速 + `translation.use_cache` 配置控制）；新增 `--output/--target-lang`。

### 2.3 OCR 引擎配置化（顺带修复审查发现）

核心管线按 `config.yaml` 的 `ocr.provider/lang/use_gpu` 构建引擎（`_build_ocr_provider`）：缺省 `easyocr`（与 README 一致）；配置的引擎导入失败回退 easyocr 并告警；完全不可用时 `_NullOCRProvider` 兜底（管线其余阶段照常）。此前 GUI 硬编码 `EasyOCR(en)`，`ocr.lang` 对 GUI 无效。

---

## 3. 路线图 #2：质检闭环接入管线

**背景**：`QualityOrchestrator` 已有完整的报告聚合/重试机制，但 GUI 管线只在跑完后**生成报告，不据此行动**——工具建好了，环没有闭上。

### 3.1 闭环流程（`core/pipeline.py` `_quality_retry_loop`）

```
管线跑完 → 聚合 render_debug.json 生成质检报告并保存
        → 筛选可自动重试的页（_select_retry_candidates）
        → 逐页：LLM 逐气泡重试失败翻译（复用 retry_page_failed_bubbles）
              → 同一渲染服务重渲染（render_debug.json 随新渲染更新）
        → 重新聚合、保存报告
        → 产出对比消息，如「质检: 自动重试 3 页，合格率 85% → 96%」
```

### 3.2 保守重试策略（设计关键取舍）

**只自动重试「内存中存在 failed 翻译气泡」的页**：

- 翻译失败（LLM 超时/限流/解析失败）→ 重试有真实增益 → 自动处理
- 纯渲染质量问题（模糊/残留/排版低分）→ 重渲染是确定性的、重试无增益 → **明确不浪费 LLM 调用**，消息标注"建议人工复核"，进入质检报告的人工审核队列

该策略由 `tests/test_services/test_quality_loop.py` 的 6 个用例锁定（failed 选中 / 渲染问题转人工 / 超重试上限排除 / 无内存结果跳过 / 健康页忽略 / 空文本对话不计入）。

### 3.3 协同修复

- **运行记录后置保存**：`_save_run_record` 移到质检闭环之后，重试产生的最终译文/渲染路径一并入档（原先存的是中间状态）
- **质检聚合跟随运行输出目录**：`QualityOrchestrator(output_dir=Path(output_dir).parent)` + `rendered_dir=Path(output_dir).name`——自定义渲染目录不再聚合到 `data/output/rendered` 的旧数据
- **取消尊重**：重试循环检查 `cancel_event`，GUI 取消按钮可中断
- **GUI 步骤条**：5 步 → 6 步（检测 → OCR → 翻译 → 渲染 → **质检** → 完成）；`quality` 阶段未识别时安全降级不变

---

## 4. 环境修复与验证

### 4.1 环境

- `.venv310` 的 **Pillow 安装损坏**（`PIL/__init__.py` 丢失，`import PIL.Image` 报 "unknown location"）→ 强制重装修复；修复前测试套件连收集都失败（18 errors）

### 4.2 验证矩阵

| 验证项 | 结果 |
|--------|------|
| 全量测试套件 | 134 通过（修复前基线）→ **147 通过**（新增 13 个用例） |
| 新增测试 | `test_core_pipeline.py`（页码范围边界 6 个 + 进度持久化闭环 1 个）、`test_quality_loop.py`（候选选择 6 个） |
| 行为验证脚本 | 7 项关键修复逐一验证（序号剥除、引号方向、TM 项目隔离、OCR 映射、限速器、翻译失败留空、fallback parse） |
| 脱机端到端 ×3 | 合成图完整管线产出 `page_N_translated.png`；页码范围模式只处理目标页且编号正确；质检闭环接入后健康运行不误触发重试、报告正确落盘 |
| 语法/静态 | `compileall` 全绿；改动文件 ruff `F,E9` 全绿 |
| **真实 GUI 启动验证**（追加轮） | `create_app()` 构建成功（461 组件/94 事件）+ 服务实启 HTTP 200 + 内置浏览器逐页核验；发现并修复 2 个启动/首屏问题（见 §4.4） |

> 注：全仓 `ruff check` 报约 2100+ 条（CI 的 lint 门禁配置过严——ANN/PL/TRY 等全量要求，改动前即红），非本次引入；本会话仅保证改动文件无 F（未用导入/未定义名）/E9（语法）类问题。

### 4.4 追加轮：真实 GUI 启动验证发现的问题（用户质疑触发）

用户质疑"GUI 能正常启动吗"后做了完整启动验证（进程内构建 + 实启服务 + 内置浏览器逐页走查），抓到两个此前所有离线验证都测不到的问题：

1. **启动崩溃（我引入的）**：`gr.update()` 返回 `dict` 不支持 `.visible` 属性访问，history_page 预启动赋值会在构建历史页时抛 `AttributeError`。用户在 VSCode 以 17:51 的旧代码启动即命中此错。修复：`dict.get("visible")`，随后整个模式被 §4.4.2 取代。
2. **Gradio 6 预启动赋值失效（历史遗留，4 个页面）**：Gradio 6 中「Blocks 构建后、启动前」对组件 `.value`/`.visible` 赋值**不会进入序列化给前端的配置**——表格首屏永远是空壳（数字列头 0..N、无数据行），点刷新按钮才出数据。历史（我复制进去的）与模型库/质检报告/评估面板（早已存在）共 4 页中招。修复：**首屏数据在组件构造时以 `value=` 传入**；浏览器实测历史页首屏直接显示 22 条记录、模型库 6 个模型、质检报告摘要（截图确认）。

> 经验教训：`import` 成功 ≠ GUI 能启动；`create_app()` 构建成功 ≠ 首屏正确。GUI 改动必须实启 + 浏览器走查。

### 4.3 验证过程抓到的回归

- e2e 脚本暴露 CLI 消费端误用 `p.page`（字段实际为 `current_page`）——及时修复
- 大段代码重排脚本误删 CLI `init`/`list_projects` 命令——从 git 历史恢复并验证 4 个命令齐全

---

## 5. 已知问题（审查发现、报告但未修——属于功能开发/设计取舍）

1. **DI 容器死线**：`TranslationService`/`QAService`/`OCRService` 子树构造后无消费者；`GemmaQAProvider.check()` 是 `NotImplementedError` 存根；`Settings` 的 `ocr_provider`/`log_level` 等环境变量覆盖字段无读取方（`CT_OCR_PROVIDER=easyocr` 静默无效）；`utils/logging.py` 零导入方；`services/ocr_health_checker.py` 零导入方。建议下个迭代决定**接线或删除**。
2. **`PageSummaryService` 永不执行**：`TranslationService.translate_page` 里的"fire and forget"是字面意义的 `pass`，跨页摘要上下文（最近 3 页）永久为空。
3. **语义 TM Phase 3 是 TODO 存根**：`_find_semantic` 返回 `[]`，且 DI 从未装配 embedding provider（`embedding_base.py` 已就绪）。
4. **质检"合格"口径不统一**：`aggregate_report` 的 `passed` 按 `overall_score`，`auto_retry_pages` 按 `not needs_retry`——两者对同一页可能给出相反结论（本次消息措辞已避开口径混用，但定义本身未统一）。
5. **`Bubble.mask` 仍是全页嵌套列表**：已二值化（0/1 整数）降约 4 倍内存，但未改为 bbox 裁剪的 ndarray（涉及序列化契约与多个消费方，需单独评估）。
6. **CLI 无 `--input-dir` 直跑模式**：必须先 `init` 注册项目才能 `run`。

---

## 6. 路线图状态

| # | 方向 | 状态 |
|---|------|------|
| 1 | 收敛双管线（core 包 + CLI 完整流程 + 真实 resume） | ✅ 本次完成 |
| 2 | 闭环质检（自动重试 + 前后对比） | ✅ 本次完成 |
| 3 | 激活角色一致性（说话人归属 → 自动填充 CharacterMemory → 注入 prompt → embedding 激活语义 TM） | ⬜ 下一步 |
| 4 | 日漫一等公民路径（MangaOCR + **竖排排版**（硬缺口）+ ja→zh 黄金路径基准） | ⬜ |
| 5 | 分发与冷启动（`comic-translator setup` 一键模型引导 / Docker） | ⬜ |
| 6 | 训练资产社区化（mixed_bubble 数据集 + YOLO 权重发布 HF） | ⬜ |
| 快赢 | 条漫长图切分、导出 CBZ/PDF、GUI 英文化 | ⬜（已验证当前均缺失） |

---

## 7. 提交建议

工作区改动横跨三个主题，建议分三个 commit：

1. **fix+chore**: 全库审查修复（§1 全部）+ 环境修复 + 测试基线
2. **refactor(arch)**: 管线收敛（§2，`core/` 新包 + CLI 重写 + 兼容层）
3. **feat(quality)**: 质检闭环（§3）

`CHANGELOG.md` 的 Unreleased 段已按同样结构记录（架构收敛 / 质检闭环 / Web UI / 修复 / 代码质量）。
