# MangaFlow Design System v1（提案）

> 阶段：Phase 2 - Design System（提案，待审核）
> 目标：冷静 / 专业 / 技术型 / 创作工具感。禁止 AI 紫蓝渐变、满屏卡片、炫技动画、emoji 图标。

---

## 1. Design Tokens

### 1.1 Color

**基调：浅色专业工具（冷静、低饱和）。深色作为可选的系统偏好模式（后续）**

| Token | 值 | 用途 |
|---|---|---|
| `--color-bg` | `#f4f4f5` (zinc-100) | 全局背景 |
| `--color-surface` | `#ffffff` | 卡片/面板/输入 |
| `--color-surface-elevated` | `#fafafa` | 悬浮层 |
| `--color-border` | `#e4e4e7` (zinc-200) | 边框/分隔线 |
| `--color-border-strong` | `#d4d4d8` (zinc-300) | 强调边框（focus） |
| `--color-primary` | `#0d9488` (teal-600) | 主操作/选中/链接 |
| `--color-primary-hover` | `#0f766e` (teal-700) | 主操作 hover |
| `--color-primary-soft` | `rgba(13,148,136,0.10)` | 选中态淡底 |
| `--color-text` | `#18181b` (zinc-900) | 主文字 |
| `--color-text-secondary` | `#52525b` (zinc-600) | 次级 |
| `--color-text-muted` | `#a1a1aa` (zinc-400) | 弱化 |
| `--color-success` | `#16a34a` | 成功 |
| `--color-warning` | `#d97706` | 警告 |
| `--color-error` | `#dc2626` | 错误 |
| `--color-info` | `#2563eb` | 信息 |

> 强调色选 **teal（青玉蓝绿）**：沉稳、专业、技术感，区别于 AI 紫蓝，也不会像亮 cyan 那样电竞。主文字用 zinc-900 而非纯黑。

### 1.2 Typography

| 角色 | 字体 | 字号 | 字重 |
|---|---|---|---|
| Display/App Title | sans | 20-22px | 700 |
| Page Title | sans | 18px | 600 |
| Section Title | sans | 14px | 600 |
| Body | sans | 13px | 400 |
| Label/Form | sans | 12px | 500 |
| Caption/Muted | sans | 11px | 400 |
| Mono (log/code/num) | mono | 12-13px | 400 |

- Sans：`Geist` → 回退系统（含 PingFang/雅黑）
- Mono：`Geist Mono`/`JetBrains Mono` → 回退 `ui-monospace`
- **全站字号只来自这套 scale，禁止页面自定**

### 1.3 Spacing（4 基数 scale）

`4 / 8 / 12 / 16 / 20 / 24 / 32`

- 页面 padding：`16px`
- 卡片内 padding：`12px`
- 控制（input/button）高度：`32px`（主 CTA 可 36px）
- 组件间距：`8px`；分区间距：`16px`
- 表格行高：`28px`
- **禁止出现 margin:13px / padding:17px 这类随机值**

### 1.4 Radius / Border / Shadow

| Token | 值 |
|---|---|
| radius | 4px（控件）/ 6px（卡片/面板）/ 0（避免过度圆角） |
| border | 1px `--color-border` |
| shadow | 仅 elevation：`0 1px 2px rgba(0,0,0,.04)`；`0 4px 12px rgba(0,0,0,.06)`（hover） |
| divider | `1px --color-border`，不滥用 `border-b` 每行 |

> 原则：**靠 spacing + typography + border + hierarchy 表达层级，不靠大卡片/大阴影/大圆角。**

---

## 2. Components

### 2.1 Navigation（Shell）

- **Sidebar**：左固定栏（~220px），分组：
  ```
  WORKSPACE     翻译
  PROJECT       项目 / 历史
  TOOLS         调试 / Render Lab / 质检 / 模型对比 / 性能
  ML LAB        标注 / 数据集 / 训练 / 模型库 / 评估
  ─────────
  SYSTEM        设置
  ```
  选中态：`--color-primary` 文字 + 淡底 + 左侧 3px 边条；hover 浅灰底。
- **Header**（顶栏）：左侧当前页标题；右侧当前项目选择器 + 运行状态指示。
- 不使用嵌套 Tabs；主内容区 = Sidebar 选中项对应的页面。

### 2.2 Buttons

| 变体 | 样式 |
|---|---|
| Primary | teal 实底 + 白字 + hover teal-700 |
| Secondary | 白底 + 边框 + hover 浅灰 |
| Ghost | 透明 + hover 淡底（用于工具按钮） |
| Danger | 红底/红字 |
| Icon | 32px 方钮，无边框 |

统一 32px 高、4px 圆角。**不做大按钮（除真正主 CTA 可 36px）。**

### 2.3 Form

Input / Dropdown / Textbox：白底 + 1px border + focus 时 `--color-border-strong` + 2px 淡 teal 环。Label 12px muted。placeholder 不替代 label。

### 2.4 Content

- **Card/Panel**：surface 白底 + 1px 边框 + 6px 圆角；hover 可 +elevation shadow。
- **Table（Dataframe 替换/样式化）**：紧凑行高 28px、表头 muted 12px、数字 mono、斑马纹可选。
- **Log Viewer**：mono 13px、行距 1.4、深色反色可选、级别前缀（INFO/WARN/ERR 文本，非 emoji）。
- **Status Badge**：文本 + 语义色点（仅真实状态时用点），success/warning/error/info。
- **Empty State**：居中图标（Phosphor 线性）+ 一句话 + 主操作按钮。
- **Loading**：骨架/进度条（细线 teal），非圆形 spinner。
- **Error State**：内联红字 + 建议操作。

### 2.5 AI Workflow（重点）

- **Pipeline Steps**：横向步骤条 `Detection → OCR → Translation → Inpaint → Typography → QA`，当前步高亮，完成步打勾（用勾号，不用 emoji）。
- **Model/Provider Selector**：下拉统一，值带说明。
- **Progress**：细线 teal 进度条 + `第 X/Y 页 · 当前阶段` 文本。
- **Job Status**：运行中（teal 脉动点）/ 成功（绿）/ 失败（红）/ 等待（灰），文本为主。

---

## 3. Shell 结构提案（Phase 3 落地）

```
┌──────────────────────────────────────────────────────┐
│ Header: MangaFlow · [当前页]          Project ▾  [状态]│
├──────────┬───────────────────────────────────────────┤
│ Sidebar  │  Main Content                             │
│ WORKSPACE│                                           │
│  翻译     │                                           │
│ PROJECT  │                                           │
│  项目/历史│                                           │
│ TOOLS    │                                           │
│  调试/    │                                           │
│  RenderLab│                                          │
│  质检/对比│                                           │
│ ML LAB   │                                           │
│  标注/数据集/训练/模型/评估                            │
│ ───────  │                                           │
│ SYSTEM   │                                           │
│  设置     │                                           │
└──────────┴───────────────────────────────────────────┘
```

> 实现方式（Gradio 可行方案）：`gr.Row` 外层，左侧 `gr.Column(scale=0.25, elem_classes="mf-sidebar")` 用按钮组模拟导航 + `gr.Column(scale=1, elem_classes="mf-main")` 放各页组件；通过按钮点击切换可见性（`gr.update(visible=...)`）或 `gr.Tabs` 但用 CSS 伪装成侧栏。**具体以 Phase 3 试做后由你验收。**

---

## 4. Translation Workspace 布局提案（Phase 4 标杆）

```
┌────────────────────────────────────────────────────┐
│ Header 卡片（轻）                                    │
│  项目/章节/输入目录  |  源语言 → 目标语言  |  输出目录 │
├───────────────────────────┬────────────────────────┤
│ 左栏（控制）                │ 右栏（预览 - 核心区）    │
│  Pipeline Steps 横向       │  Image 预览（大）       │
│  Model/Provider 选择       │  + 页导航（上一张/下一张）│
│  [Translate] 主按钮        │                        │
│  进度条 + 当前页文本        │                        │
│  高级选项（折叠）            │                        │
├───────────────────────────┴────────────────────────┤
│ 底部：日志（mono，可过滤）+ 状态/性能行                │
└────────────────────────────────────────────────────┘
```

- 预览是核心区（占 2/3），不是普通卡片。
- Pipeline 步骤可视化是"我正在哪一步"的关键。
- 主按钮 `Translate` 只有 1 个，明确。

---

## 5. 实施顺序（Phase 3-5，等你审核）

1. **P3 Shell**：建 `gui/styles/tokens.css + components.css + layout.css`（从 app.py 拆出）；Sidebar + Header + 全局 layout；迁移 app.py 结构。
2. **P4 Translation Workspace**：按上面布局重做 pipeline_page（仅视图层，事件逻辑不动）。
3. **P5 其余页面**：按 Audit 优先级逐页迁移到 Design System。
4. **每阶段**：你启动 GUI 验收截图；跑 pytest 保功能不回退。

---

*Phase 2 提案结束。等待审核后进入 Phase 3 代码实现。*
