# MangaFlow GUI Layout Stability Audit

> 阶段：Phase 3B Stability Gate（只读审计，未改任何代码）
> 依据：实际代码逐行分析 `compare_page.py` / `render_lab_page.py` / `benchmark_page.py` + `app.py` 全局 CSS + 相关 service。
> 目标：找出 Layout Shift 的真实触发点，区分"真 bug"与"正常内容增长"，提出**统一修复模式**，供 Phase 3B Stability Fix 实施。

---

## 1. Problem Matrix

| Page | Component | Trigger | Layout Shift | Severity | Root Cause |
|------|-----------|---------|--------------|----------|------------|
| Model Compare (`compare_page.py`) | `gr.Image(height=400)` ×4 | 图片从空 → 加载（点"开始对比"后） | **YES** | **High** | Gradio `Image(height=400)` 仅设占位高；图片加载后按原图 aspect ratio 重算实际显示高度 → 下方 `gr.Markdown` info 行 + 整个 Row 下移 |
| Model Compare | `gr.Image(label="上传图片", height=300)` | 上传图片 | YES | Medium | 同上：300px 占位 vs 实际图高 |
| Model Compare | `gr.Markdown("")` (info ×4) | 空 → 有内容（`**N** 个区域…`） | YES | Low | 空 Markdown 高度≈0 → 有文字后撑高几行（正常内容增长，但叠加在图片 shift 上放大位移） |
| Render Lab (`render_lab_page.py`) | `gr.Image(height=450)` ×4 | 上传/加载图片 | **YES** | **High** | 同 Model Compare：450px 占位 vs 原图高 |
| Render Lab | `gr.Image(height=350)` ×2 (A/B) | 点 A/B 对比 | YES | Medium | 同 |
| Render Lab | `scores_md` / `ab_scores_md` `gr.Markdown("**…未运行**")` | 运行后变多行 | YES | Low | Markdown 行数变化（正常增长，但无预留空间） |
| Evaluation (`benchmark_page.py`) | `gr.Dataframe(summary/detail)` | 数据从 `[]` → 多行 | YES | **Medium** | 两个 Dataframe 初始 `value=[]`（高度≈表头），reload 后填充多行 → 高度增长。且页面加载时 `reload()` 被直接调用（非事件），初始值可能已非空，仍可能变化 |
| Evaluation | `wrap=True`（两个 Dataframe） | 列内容长时换行 | YES | Medium | `wrap=True` 让长文本换行 → 行高/表高不可预测 → 撑高 |

---

## 2. Root Cause（统一）

### 核心根因 A：`gr.Image(height=<px>)` 的"占位高度 ≠ 实际渲染高度"

Gradio 的 `gr.Image(height=...)` 只设定**初始容器占位高度**。图片加载后，Gradio 前端按**图片自身宽高比**重新计算 `<img>` 的显示尺寸（`object-fit` / 容器 resize）。因此：

```
初始：Image 容器 = 400px 占位（空/无图）
加载后：Image 容器 = 实际图高（如 800px×600px → 显示高 ~600px）
```

→ 图片下方的 `gr.Markdown` info、下一 Row、整个页面内容**整体向下位移**。

**三个页面（Compare / Render Lab）都用了这个模式**，是同一根因，不是各自独立 bug。

### 核心根因 B：动态内容无预留空间

- `gr.Markdown("")` 初始高度 ≈ 0，有内容后撑高（info 行、scores）
- `gr.Dataframe(value=[])` 初始高度 ≈ 表头，填充后增长（Evaluation）
- 这些是"正常内容增长"，但**叠加在根因 A 的图片位移上**，放大了 Layout Shift 观感。

### 核心根因 C：`wrap=True` 表格

Evaluation 两个 Dataframe `wrap=True` → 长文本（"标签分布"、合成 metrics 字符串）换行 → 行高不可预测 → 表高动态变化。

### 排除项（明确不是根因）
- 无 `gr.update(visible=...)` 在这些页面（Compare/Render Lab 均无 visible 切换；Evaluation 无）
- 无嵌套 Accordion（这些页面无 Accordion）
- 无 CSS transition 改尺寸（app.py 全局 CSS 无 `height` transition）
- 无 `transform: scale()`（全局仅 `.mf-nav-btn:active` / `#mf-start-btn:active`，与这些页面无关）
- app.py 全局 CSS 对图片/表格**无高度规则**（唯一 `.mf-preview { min-height: 420px }` 只在 Translation Workspace，不涉此三页）

---

## 3. Recommended Fix

### 统一模式：`StablePanel`（稳定容器 + 预留空间）

对**所有含动态 `gr.Image` 的页面**，用**固定高度的容器包裹图片**，让图片加载前后容器高度一致：

```python
# 统一 helper（放在 gui/components/ 或各页复用）
def stable_image(label: str, height: int = 360) -> gr.Image:
    """Layout-stable 图片组件：
    Gradio 的 height 只设占位；为稳定布局，把图片放进固定高度容器。
    实现：用 gr.Image + elem_classes='mf-img-stable'，配合全局 CSS 固定容器高度 + object-fit: contain。
    """
    return gr.Image(label=label, height=height, elem_classes="mf-img-stable")
```

全局 CSS（app.py `_CSS`，Phase 3B Stability Fix 时加）：

```css
/* 稳定图片容器：高度固定，图片等比 contain，不因加载改变容器尺寸 */
.mf-img-stable {
  height: 360px !important;
  overflow: hidden;
}
.mf-img-stable img {
  object-fit: contain !important;
  width: 100% !important;
  height: 100% !important;
}
```

**为什么能解决**：容器高度固定为 360px，图片无论宽高比都 `object-fit: contain` 填充，不撑开父容器 → 下方内容**不位移**。

---

### 各页面具体方案

#### Model Compare（compare_page.py）
| 问题 | 修复 |
|---|---|
| `gr.Image(height=400)` ×4 | 改用 `stable_image(height=360)`（`mf-img-stable`），四图等权 2×2，容器高度统一 |
| 上传图 `height=300` | 同样 `mf-img-stable`，固定 280px |
| info `gr.Markdown("")` | 预留固定高度：用 `elem_classes="mf-info-line"` + CSS `min-height: 22px`（一行高），始终占位，内容变化不撑高 |

#### Render Lab（render_lab_page.py）
| 问题 | 修复 |
|---|---|
| `gr.Image(height=450)` ×4 | `stable_image(height=360)` |
| A/B `height=350` ×2 | `stable_image(height=280)` |
| `scores_md` / `ab_scores_md` | 预留 `min-height`（如 24px / 48px），或保持组件常驻（本来常驻，只是内容行数变 → 用固定 min-height 吸收） |

#### Evaluation（benchmark_page.py）
| 问题 | 修复 |
|---|---|
| 两个 Dataframe 动态高度 | **方案 C：固定表格区域** —— 给 Dataframe 容器 `max-height` + `overflow: auto`（CSS `.mf-table-fixed { max-height: 420px; overflow-y: auto; }`），表头固定、数据区滚动；或统一 `.mf-table` 加 `max-height` |
| `wrap=True` | 改 `wrap=False`（长文本截断），避免行高不可预测 |

---

### 统一修复模式总结（给后续所有页面复用）

```
Layout-Stable Panel（3 条铁律）：
1. 动态图片 → 固定高度容器 + object-fit: contain（mf-img-stable）
2. 动态 Markdown/Dataframe → 预留 min-height / max-height（mf-info-line / mf-table-fixed）
3. 表格 → wrap=False + max-height + overflow:auto，禁止长文本撑高
```

这 3 条以**全局 CSS class** 形式加进 `app.py _CSS`（一次定义，所有页面复用），页面侧只改 `gr.Image(..., elem_classes="mf-img-stable")` 等。**不引入 JS / setTimeout / 绝对定位 / transform**（符合 brief 方案 E）。

---

## 4. 是否需要改 app.py？

- **需要**：加 3 个全局 CSS class（`mf-img-stable` / `mf-info-line` / `mf-table-fixed`）。这是**设计系统统一部分**，不是页面 hack。
- **不需要**：改 Shell / Header / Sidebar / 业务逻辑。

---

## 5. 优先级

| 优先级 | 页面 | 内容 |
|---|---|---|
| P0 | Model Compare + Render Lab | 图片 Layout Shift（最高观感影响，同一根因） |
| P1 | Evaluation | 表格动态高度 + wrap |
| P2 | 全局 CSS class 定义（一次） | `mf-img-stable` / `mf-info-line` / `mf-table-fixed` |

---

## 6. 明确区分

| 现象 | 分类 | 处理 |
|---|---|---|
| 图片加载后下方内容下移 | **真 Layout Shift（修复）** | `mf-img-stable` 固定容器 |
| Accordion/表格数据增多 | 正常内容增长（可接受） | 若需稳定 → 预留空间 |
| 图片本身按比例显示 | 正常（保留） | `object-fit: contain` 保留原比例 |
| 页面打开瞬间从无到有图 | 真 Layout Shift（修复） | 固定容器 |

---

*Audit 结束。未修改任何代码，未 commit。等待审核后进入 Phase 3B Stability Fix。*
