# Comic Translator 项目改进计划

> 基于代码审查和实际修改经验整理

---

## 状态标记

- ✅ **已完成** — 本轮修复已做完
- 🔧 **进行中** — 部分完成，需继续
- 📋 **待开始** — 规划中
- 💡 **建议** — 未来方向

---

## 阶段一：架构修复（已完成）

### 1.1 删除 LangGraph
- ✅ 删除 `workflow/`、`nodes/` 目录和 `tests/test_workflow/`
- ✅ 从 `pyproject.toml` 移除 `langgraph` 依赖
- **原因**: 8 个节点全是空壳，从未被执行，与异步 Service 层不兼容

### 1.2 统一 DI 容器
- ✅ `utils/di.py` 改为从 `config.yaml` 动态读取配置
- ✅ 新增 `create_llm_client_from_config()` 支持 ollama/api 模式
- ✅ 新增 `create_ocr_provider_from_config()` 支持 paddle/easyocr/manga_ocr
- ✅ 新增 `create_translation_provider_from_config()` 自动选择翻译引擎
- ✅ 新增 `_resolve_env_var()` 支持 `${VAR_NAME}` 环境变量语法
- ✅ 修复 `RenderService` 构造参数错误（之前传了不存在的 `provider` 参数）
- **原因**: 旧代码硬编码了模型和客户端，不读 config.yaml

### 1.3 重构 GUI 翻译逻辑
- ✅ 从硬编码 HTTP 改为统一 LLMClient 调用
- ✅ 双路径设计：优先 DI 容器，失败时自动回退到 HTTP
- ✅ 翻译结果应用逻辑抽取为 `_apply_translation_results()` 复用
- **原因**: 旧代码绕过所有 Service 抽象层，直接 HTTP 请求

### 1.4 实现存储层
- ✅ `ProjectRepository` 完整 CRUD（save/find_by_id/find_all/delete/update_progress）
- ✅ `TranslationMemoryRepository` 增强（find_similar/update_usage/count/delete）
- ✅ 新增 `GlossaryRepository`
- **原因**: 旧代码全是 TODO 占位

### 1.5 实现 CLI 命令
- ✅ `init` — 创建项目，自动计算图片数，保存到 SQLite
- ✅ `run` — 使用 `BatchTranslationService` 运行完整翻译管线
- ✅ `resume` — 从断点继续
- ✅ `list-projects` — 列出所有项目
- **原因**: 旧代码全是 TODO

### 1.6 配置清理
- ✅ `config.yaml` 恢复 API key，添加 `${VAR}` 环境变量支持
- ✅ `config/settings.py` 修复 `pydantic_settings` 的 `yaml_file` 警告
- ✅ `.gitignore` 修复 `uv.lockconfig.yaml` 笔误
- ✅ 创建 `examples/` 目录，整理 demo 脚本
- **原因**: 明文 API key 安全风险 + 配置混乱

---

## 阶段二：功能优化（待开始）

### 2.1 术语表 + 角色记忆接入 GUI 翻译
- ✅ **已完成** | 优先级: P0 | 工作量: 2 小时 | 价值: 翻译质量明显提升

**现状（已完成）**: `_translate_dialogues` 通过 `_build_translation_context()` 从 DI 容器获取 glossary + character identity，拼入 system prompt；翻译后 `_save_auto_glossary()` 自动回写术语表。
- 代码位置: `gui_pipeline_service.py` 的 `_build_translation_context()` / `_save_auto_glossary()` / `_translate_dialogues()`

### 2.2 OCR 后处理接入 GUI
- ✅ **已完成** | 优先级: P0 | 工作量: 1 小时 | 价值: OCR 准确率提升

**现状（已完成）**: GUI 的 OCR 步骤后调用 `ocr_postprocess_service.process_page()`（`_run_async` 中 OCR 之后）。
- 代码位置: `gui_pipeline_service.py` 的 `self.ocr_postprocess`（init）+ `_run_async` OCR 步骤后

### 2.3 翻译错误反馈
- 📋 **优先级: P1 | 工作量: 半天 | 价值: 用户体验提升**

**现状**: 翻译失败时（API 429、key 无效、网络超时），用户看不到任何提示，只在渲染图上看到 "TRANSLATION FAILED"。

**改法**: 
- `PipelineProgress` 增加 `error` 字段
- `_run_async()` 的 yield 循环中检查错误并 yield 错误信息
- GUI 前端显示 toast 或 alert

### 2.4 配置统一化
- 📋 **优先级: P1 | 工作量: 2 小时 | 价值: 维护成本降低**

**现状**: 配置分散在 `config.yaml`、`defaults.yaml`（未正确加载）、`gui_pipeline_service.py`（硬编码）。

**改法**:
- 删掉 `defaults.yaml`（已经被 `Settings` 类忽略加载）
- 所有配置统一从 `config.yaml` 读取
- 模型路径从 `config.yaml` 读取，消除硬编码

### 2.5 端到端集成测试
- 📋 **优先级: P1 | 工作量: 半天 | 价值: 质量保障**

**现状**: 42 个单元测试全部通过，但没有测试验证整条流水线能跑通。

**改法**: 一张测试图 → `GuiPipelineService.run_pipeline()` → 检查输出文件存在且翻译不为空。

### 2.6 翻译提速（✅ 已完成）
- ✅ **并发 + 去固定 sleep**：GUI 翻译 worker 数从硬编码 2 改为 `translation.workers`（默认 4），删掉每页 `sleep(0.5)` 与 CLI 每批 `batch_delay=1.0`（改默认 0，`--batch-delay` 可调）；并发节奏交给 LLM 客户端令牌桶（rps）与 429 退避控制。
- ✅ **翻译记忆 + 页面翻译缓存复用**（GUI）：`_translate_dialogues` 翻译前先查语义 TM（精确命中）与页面级翻译缓存，命中直接复用跳过 LLM；翻译后回写 TM 与缓存（`CacheService`，键 `{project}:translation:{page}`）。重跑几乎免费，部分改动只重翻变化行。
- ✅ **回归测试**：新增 `tests/test_services/test_translation_reuse.py`（3 例），全套 124 测试通过。
- ✅ **渲染图片名排序修复**：`collect_images`（GUI 与 CLI 批处理）改用自然排序（`natural_sort.py`，`2.jpg < 10.jpg`），页码始终=阅读顺序且与翻译/重译顺序无关（重译只覆盖同名文件，不会编号错乱/反转）。新增 `tests/test_utils/test_natural_sort.py`。
- 🔧 **未做**：OCR 缓存接入（重跑可进一步跳过 GPU 检测/OCR，需缓存整页 pre-render 结果，改动较大，列为后续）。

---

## 阶段三：新功能开发（建议）

### 3.1 多语言支持
- 💡 **优先级: P2 | 工作量: 3 天 | 价值: 用户群扩大 10 倍**

**现状**: 只支持 `en → zh`。

**改法**:
- 添加日语支持（`ja → zh`）：MangaOCR + 日语 PromptBuilder
- 添加韩语支持（`ko → zh`）
- 添加中文出海（`zh → en`）
- `BatchPromptBuilder` 和 `PromptBuilder` 子类需要语言参数

**日漫汉化是最大的市场，工作量不大但用户群会大很多。**

### 3.2 交互式翻译编辑
- 💡 **优先级: P2 | 工作量: 1 周 | 价值: 从"玩具"到"工具"**

**现状**: 用户只能接受翻译结果或重新跑一遍。

**改法**:
- 在调试查看器页面增加文本编辑框
- 编辑后调用 `render_service.render_page()` 只重新渲染该块
- 编辑后的结果保存到翻译记忆

### 3.3 翻译置信度打分
- 💡 **优先级: P2 | 工作量: 2 天 | 价值: 用户知道哪些翻译可靠**

**现状**: QA 只有 pass/fail 二值判断。

**改法**:
- 每个 DialogueBlock 一个置信度分数（0-100）
- 低分块在 GUI 中高亮显示
- 自动重跑低分块

### 3.4 批量导出
- 💡 **优先级: P3 | 工作量: 2 天 | 价值: 可用性提升**

**现状**: 输出是一堆 PNG 文件。

**改法**:
- 导出为 PDF
- 导出为 CBZ/CBR（漫画阅读器格式）
- 导出为带字幕层的格式

### 3.5 增量翻译
- 💡 **优先级: P3 | 工作量: 3 天 | 价值: 大型项目可用**

**现状**: 每次全量跑。

**改法**: 检测已翻译页面，只处理新增/修改的页面。

### 3.6 本地模型优先
- 💡 **优先级: P3 | 工作量: 1 周 | 价值: 降低上手门槛**

**现状**: 依赖云端 API。

**改法**: 优先使用本地 Ollama 模型，云端 API 作为 fallback。

### 3.7 模型自动下载
- 💡 **优先级: P3 | 工作量: 1 天 | 价值: 新用户上手时间从 1 小时降到 5 分钟**

**现状**: 模型文件要用户手动下载放到 `models/` 目录。

**改法**: `python -m comic_translator download-models` 命令，自动从 HuggingFace 下载。

### 3.8 排版用 AI 替代规则引擎
- 💡 **优先级: P3 | 工作量: 3 天 | 价值: 排版质量上限提升**

**现状**: 6 个 Solver 的规则引擎，效果上限低。

**改法**: 让 LLM 直接输出排版参数（字号、行坐标、对齐方式），替代 `FontSizeSolver` + `BalancedLineBreaker` + `ShapeLayoutSolver`。

### 3.9 附带微型 demo 数据集（让 ML 模型开发功能开箱可见）
- 💡 **优先级: P3（上架前可选） | 工作量: 1~2 小时 | 价值: reviewer 能实际点通标注/训练**

**背景**: 真实标注数据在 `dataset/`（mixed_bubble 等，已被 gitignore 不入仓库）；
clone 后 GUI 的「ML 模型开发」（数据集/标注/训练中心）会显示无数据集，容易被误判为功能缺失/坏掉。

**改法**:
- 在 `examples/` 放一个微型 demo 数据集（5~10 张图 + YOLO 标签，几 MB，需确认图片许可）
- 提供 `python scripts/import_demo_dataset.py` 一键导入到 `dataset/demo`
- README 快速开始可选提一句「想体验训练/标注？先 import demo 数据集」

---

## 四、实施路线图

```
现在
 │
 ├── 这周（阶段二）
 │   ├── P0: 术语表 + 角色记忆接入 GUI 翻译（2h）
 │   ├── P0: OCR 后处理接入 GUI（1h）
 │   └── P1: 端到端测试（半天）
 │
 ├── 下周
 │   ├── P1: 翻译错误反馈（半天）
 │   ├── P1: 配置统一化（2h）
 │   └── P2: 日漫汉化 ja → zh（3天）
 │
 ├── 下个月
 │   ├── P2: 交互式翻译编辑（1周）
 │   ├── P2: 翻译置信度打分（2天）
 │   └── P3: 批量导出（2天）
 │
 └── 未来
     ├── 本地模型优先（1周）
     ├── 增量翻译（3天）
     ├── 模型自动下载（1天）
     └── AI 排版（3天）
```

---

## 五、关键决策点

### 现在就要选

1. **LangGraph 已删**，确认是否保留当前纯 Python 编排方式
2. **GUI 和 Batch 两条管线是否合并** — 建议合并，代码量减半

### 近期要决定

3. **是否支持日漫** — 这是最大的用户需求来源，也是最大的技术挑战（日语 OCR 比英语难）
4. **是否开源** — 影响代码质量标准和文档投入

### v1 发布范围（2026-08 已决策）

5. **ML 模型开发（标注/训练/评估）保留但不作为 v1 卖点**：
   - 代码保留在仓库 + GUI「🛠 高级工具」折叠区（定位正确，不删）
   - README 不宣传这组功能，只标「进阶、需自备数据」（已加）
   - v1 主打**翻译管线**；训练/标注的数据（`dataset/`）被 gitignore，reviewer 看不到
   - **demo 数据集（§3.9）后置到 v1.1**，届时让这组功能可实际点通
   - 训练功能须由本人在真实 GPU 环境跑通一次（epochs=2 冒烟）后再考虑扶正为正式卖点

### 远期考虑

5. **是否产品化** — 决定 Web 化、多用户、API 化的投入