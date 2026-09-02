# MangaFlow UI Audit v1

> 阶段：Phase 1 - Audit（仅分析，不改代码）
> 依据：当前 `src/comic_translator/gui/` 全部页面、组件、app.py、主题与 CSS 的实读。
> 定位：MangaFlow 是 **专业的 AI 漫画翻译工作站**（不是 Landing Page，不是普通 SaaS Dashboard）。

---

## 0. 当前实现全貌（事实清单）

- **入口**：`gui/app.py` → `create_app()` → Gradio Blocks + `launch(theme, css)`
- **主题**：`gr.themes.Soft(primary_hue="cyan", secondary_hue="slate", neutral_hue="slate")`
- **样式**：全部内联在 `app.py` 的 `_CSS` 字符串（无独立 styles/tokens 目录）
- **页面**：14 个（含 `__init__.py`），大小 2KB~20KB
- **组件**：仅 3 个共享组件（detection_overlay / dir_scanner / perf_format），无 UI 级共享组件库
- **当前导航（我上一轮已重构为 5 主区）**：
  ```
  翻译 / 记录(项目+历史) / 设置 / 模型(标注/数据集/训练/模型库/评估) / 高级(调试/性能/质检/对比/Render Lab)
  ```
- **组件使用量**（全页面）：
  `gr.Markdown 93` / `gr.Row 55` / `gr.Column 44` / `gr.Button 51` / `gr.Dropdown 25` / `gr.Textbox 24` / `gr.Image 17` / `gr.Dataframe 16` / `gr.Checkbox 11` / `gr.Accordion 9` / `gr.Slider 8` / `gr.State 6` / `gr.Tab 3` / `gr.Number 3` / `gr.Radio 1` / `gr.CheckboxGroup 1`

---

## 1. Information Architecture 评估

### 当前产品域划分（我对现有 14 页的分组）

| 产品域 | 页面 |
|---|---|
| **Workspace（核心工作流）** | 翻译（pipeline） |
| **Project/History（项目与记录）** | 项目管理、历史 |
| **Tools（调试/质检）** | 调试查看器、性能监控、质检报告、检测模型对比、Render Lab |
| **ML Lab（模型开发）** | 标注中心、数据集、训练中心、模型库、评估面板 |
| **System** | 设置 |

### 判断

- **方向正确**：上一轮已把 14 个平铺页面收敛成 5 个主区，普通用户路径（翻译）与开发者路径（调试/ML）已分离。
- **问题 1：主区仍是"Tab 套 Tab"**。`记录`/`模型`/`高级` 内部是嵌套 `gr.Tabs`。Gradio 的嵌套 tab 视觉上就是"套娃"，层级感弱，且之前触发过抖动（已修 transform，但嵌套本身仍显得像网页标签页而非桌面工具的导航）。
- **问题 2：缺少"上下文感"**。桌面工具用户应时刻知道"我在哪个工作区、当前项目是什么、正在处理哪一步"。现在 Tab 只告诉你"在翻译页"，不告诉你"在处理哪个项目、管的是哪个数据源"。
- **问题 3：Workspace 没有"工作台"结构**。翻译页是"表单→按钮→日志"的纵向流，不是"左工具/右预览"的工作台。

### 建议方向（Phase 3 详述）
用 **Sidebar（左导航）+ Header（项目/上下文）+ 主内容区** 的桌面工具壳替代纯 Tab 套娃。参考 brief 的 Shell 草图，但根据真实页面重组（见下）。

---

## 2. Navigation 评估

- **页面数量**：14 个，对 Gradio 来说偏多；Tabs 作为唯一导航手段（无 Sidebar、无面包屑）。
- **路径自然性**：用户从 `翻译 → 调试 → 渲染 → QA` 需要跨 2 个主 Tab（翻译 和 高级），且高级里要再切子 Tab。步骤之间的"查看上一步产物"不顺畅（Debug 与 Render Lab 都在高级，但 Pipeline 在翻译）。
- **缺状态感知**：无"当前活动页面"以外的上下文（无面包屑、无 project 选择器在全局 header）。
- **Tabs 使用**：主 tab 合理，但嵌套 tab 是"网页感"来源之一。

### 建议
- 引入 **Sidebar**：左栏分组导航（Workspace / Tools / ML Lab / System），选中态高亮；主内容区只渲染当前页。
- 引入 **Header**：当前项目、当前模式、运行状态指示。
- 这样 `翻译`（Workspace）→ `调试`（Tools）→ `Render Lab` 的路径变成"侧栏点选"，比 Tab 套娃更桌面化。

---

## 3. Visual Hierarchy 评估

当前各页面层级来源混杂：

- **标题**：用 `gr.Markdown("## 翻译")` 等，但**字号/字重/颜色没有统一 token**，每个页面自己写。
- **Section**：翻译页用 `mf-card` 分组（①输入 ②操作 ③高级 ④进度 ⑤日志），但 `mf-card` 样式是"大圆角白卡片"，与"专业工具"目标不符（应更扁平、靠边框和 spacing）。
- **Primary Action**：翻译页 `开始翻译` 是 `variant="primary"` + 大号，但**没有全局统一的按钮层级规范**（primary/secondary/ghost/danger 未定义）。
- **Status**：用 emoji（🟢⏳✅❌⚠️🔧👁️💬🎨）作为状态/阶段图标——**违反 brief 禁用项**（emoji 作为主要 UI icon），且不同页面状态色不统一（有的绿点、有的文字）。
- **Warning/Error**：文本内嵌 emoji 标记，无统一 badge/提示组件。

### 建议（Phase 2 定 token，Phase 3 落地）
标题/正文/说明/等宽建立字号层级；状态用 **文本 + 语义色**（success/warning/error/info）而非 emoji；按钮定义 primary/secondary/ghost/danger；层级主要靠 spacing+typography+border，**少用大卡片**。

---

## 4. Density 评估

- **现状**：`gradio-container` max-width 1440、padding 大；`gr.Row/Column` 用 Gradio 默认间距；`mf-card` 内 padding 0.85rem/1rem。整体偏"网页留白"。
- **问题**：专业工具（尤其 debug、dataset、history、log、table）应该**高密度**。当前 Dataframe（16 处）用 Gradio 默认行高；日志 Textbox 12 行；统计表偏散。
- **建议标准**：
  - 控制高度：输入/按钮 32-36px（非 46px 大按钮，除真正主 CTA）
  - 页面 padding：`8px` 级（spacing scale 见 Design System）
  - 表格：紧凑行高（padding 4-6px），等宽数字
  - 日志：等宽字体 + 行距 1.4 + 小号（12-13px）
  - 图像预览：作为工作台核心区（大），不是普通卡片

---

## 5. Gradio Default Look 暴露点

以下位置最"明显是 Gradio 默认"（Phase 2/3 用统一 Design System 压掉）：

| 元素 | 暴露点 | 当前 | 目标 |
|---|---|---|---|
| Tabs | 嵌套 tab 视觉 | 网页标签感 | Sidebar + 选中态 |
| Button | 51 个按钮无统一规范 | 默认灰底/primary 蓝绿 | primary/secondary/ghost/danger 层级 |
| Dropdown | 25 个 | 默认样式 | 统一 token（背景/边框/focus 环） |
| Dataframe | 16 个 | 默认表 | 紧凑表格 + 表头样式 |
| Accordion | 9 个 | 默认折叠 | 统一折叠头 + 图标 |
| Status | emoji 状态 | 混乱 | 语义色 badge / 文本 |
| Image | 17 个 | 默认占位 | 预览区（衬底/缩放控制） |
| Slider | 8 个（含进度条） | 默认滑条 | 细进度条/滑条统一样式 |
| Checkbox | 11 个 | 默认 | 统一 token |
| Layout | Row/Column 55/44 | 默认间距 | spacing scale |

---

## 6. 组件重复与缺失

- **重复**：各页面各自写 `gr.Markdown("## 标题")`、各自 emoji、各自状态格式。缺一个统一的 `page_header(title)` / `status_badge(...)` / `button_row(...)`。
- **已有可复用**：`components/detection_overlay.py`（视觉工具）、`components/perf_format.py`（性能格式化）、`dir_scanner.py`。**无 UI 组件库**。
- **缺失**（Phase 2 定义、Phase 3 建）：
  - `PageHeader`（标题 + 上下文/面包屑）
  - `StatusBadge`（语义色文本徽标，非 emoji）
  - `EmptyState`（数据集为空/无历史时的引导）
  - `PipelineSteps`（检测→OCR→翻译→修复→排版→QA 的可视化步骤）
  - 统一 `button`（primary/secondary/ghost/danger）
  - 统一表格/日志样式

---

## 7. 当前架构对 UI 的限制（重要）

| 限制 | 原因 | 推荐方案 | 风险 |
|---|---|---|---|
| **无 Sidebar 原生组件** | Gradio 无内置 Sidebar | 用 `gr.Row(scale=...)` 左侧固定栏 + 右侧内容，或 CSS 固定侧栏 | 中：布局复杂度上升，需保证窄屏可用 |
| **CSS 全内联在 app.py** | 项目无 styles/ 目录 | 拆 `gui/styles/tokens.css + components.css + layout.css`，app.py 注入 | 低 |
| **嵌套 Tabs 视觉** | Gradio 嵌套 tab 样式难压 | 用 Sidebar 替代嵌套 tab | 中：改动 app.py 结构 |
| **组件样式命中难** | Gradio 6 class 名不透明 | 用 `elem_id/elem_classes` + `gradio-container` 作用域 + `!important` 兜底 | 低（已验证注入生效） |
| **无法查看渲染结果** | 我（agent）无浏览器/截图能力 | 分阶段改，**每阶段由你验收**截图 | 依赖你反馈 |
| **业务逻辑必须不动** | brief 约束 | 只改 gui/ 视图层，不碰 services/providers/models | 需回归验证 |

---

## 8. 优先级排序（建议执行顺序）

1. **P0 建立 Design Token**（color/typography/spacing/radius/border/shadow）——一切的地基
2. **P0 重构 Shell**（Sidebar + Header + 全局 layout）
3. **P0 Translation Workspace**（作为视觉标杆，工作台式布局）
4. **P1 组件库**（PageHeader/StatusBadge/Buttons/EmptyState/PipelineSteps）
5. **P1 迁移**：Projects → History → Debug → Render Lab → Quality → ML Lab(5页) → Settings
6. **P2 打磨**：表格/日志/进度/空态/加载态/错误态

---

*Phase 1 Audit 结束。下一步：Phase 2 Design System（见 UI_DESIGN_SYSTEM.md），等你审核后再进入 Phase 3 代码。*
