# Bubble-Aware Architecture Design

## 问题分析

当前 Render 拿到的是 OCR block 的 bbox，而不是气泡区域。OCR bbox 只包裹文字本身，尺寸远小于气泡，导致中文（更长）必然溢出。

```
修复前:  OCR bbox → Layout → 中文溢出
修复后:  Bubble bbox → Layout → 中文适应
```

---

## Phase 1: Bubble Detection

### 技术方案：OpenCV 轮廓分析（无需 ML 模型）

成熟方案分析：

| 工具 | 方法 | 适用性 |
|------|------|--------|
| BallonsTranslator | 白色区域轮廓分析 + 区域生长 | ✅ 直接可复用思路 |
| MangaOCR | 基于 CRNN，不做气泡检测 | ❌ 只做文字识别 |
| PaddleOCR | DBNet 检测文本行 | ❌ 不做气泡检测 |
| ComicTextDetector | YOLO-based | ❌ 需要训练 |

结论：**OpenCV 白区域检测**是目前最成熟的无需模型方案。

### 算法步骤

```
1. 灰度化
   img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

2. 形态学闭合（合并气泡内的文字间隙）
   kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
   closed = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)

3. 自适应二值化
   binary = cv2.adaptiveThreshold(closed, 255, 
               cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
               cv2.THRESH_BINARY_INV, 31, 5)

4. 轮廓检测
   contours, _ = cv2.findContours(binary, 
               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

5. 过滤
   For each contour:
     - area > min_area
     - 非全幅边界
     - 排除 tiny/noise

6. 合并邻近轮廓
   For each pair of contours:
     if distance < threshold:
       merge → convex hull

7. 气泡类型分类
   if 顶部 y < 0.15 * img_h: narration
   elif 高宽比 < 0.4 or 尾部指向左右: speech
   else: narration / sfx
```

### 文件

```
services/bubble_detector.py
  class BubbleDetector
  @dataclass BubbleRegion
```

### 依赖

当前项目已安装：
- `opencv-python` ✅
- `numpy` ✅

**零新增依赖。**

---

## Phase 2: OCR Block → Bubble Mapping

### 方案

Simple 方法（足够有效）：

```
For each OCR block b:
  center = (b.x + b.w/2, b.y + b.h/2)
  
  For each bubble region r:
    if cv2.pointPolygonTest(r.contour, center) >= 0:
      assign b to r
      break

  if no bubble found:
    fallback: 找最近气泡（距离阈值 50px）
  
  if still no match:
    create singleton bubble(b.bbox)
```

### 输出

```
@dataclass ConversationRegion:
    id: str
    bubble_bbox: BBox
    bubble_type: str
    ocr_blocks: list[NormalizedOCRBlock]
    combined_text: str    # 同一气泡内所有 OCR 文本合并
    translated_text: str  # 翻译后的合并文本
```

### 文件

```
services/bubble_assignment_service.py
  class BubbleAssignmentService
```

---

## Phase 3: Bubble-Based Translation

### 改动

修改 `batch_prompt_builder.py`：

**当前（逐 OCR block）：**
```
[id=ocr1] Hello.
[id=ocr2] How are you?
[id=ocr3] Nice weather.
```

**改为（逐 Bubble）：**
```
=== BUBBLE 1 (speech) ===
Hello. How are you? Nice weather.
```

同一气泡内的所有文字合并为一段翻译。模型一次性输出整段翻译，上下文更完整。

### 修改文件

| 文件 | 改动 |
|------|------|
| `models/render_models.py` | 新增 `ConversationRegion` 模型 |
| `services/batch_prompt_builder.py` | `build_user_prompt()` 按 Bubble 组织文本 |
| `providers/translation/base.py` | 新增 `translate_bubble()` 接口（可选） |

### 翻译提示词改动

```
Before:
  [id=ocr11] Oh BABE!
  [id=ocr12] I Just Can't Believe
  [id=ocr13] How Great This PLACE Looks

After:
  --- BUBBLE speech ---
  Oh BABE! I Just Can't Believe How Great This PLACE Looks
```

---

## Phase 4: Bubble-Based Layout

### 改动

修改 `text_layout_service.py`：

**当前：**
```
layout(text=ocr_text, box=ocr_bbox)
```

**改为：**
```
layout(text=bubble_text, box=bubble_bbox)
```

`bubble_bbox` 是气泡边界框，比 OCR bbox 大得多。中文文本能正确排版。

### 拟合算法

```
fit_text_to_bubble(text, bubble_bbox, font_profile):
  for size from max_size down to min_size:
    for attempt in [single_line, multi_line]:
      lines = wrap_text(text, bubble_width, font, size)
      total_height = lines * line_height
      if total_height <= bubble_height:
        return (size, lines)
  return (min_size, lines)
```

### 实测预期

| 指标 | OCR bbox | Bubble bbox |
|------|----------|-------------|
| 宽度 | 280px | 400px |
| 高度 | 40px | 120px |
| 可用字号 | 12-18 | 20-36 |
| 行数 | 1-2 | 2-4 |
| 溢出 | ✅ 可能 | ❌ 不溢出 |

---

## Phase 5: Render Pipeline V5

### 新流程

```
Comic Image
  ↓
BubbleDetector.detect()
  → list[BubbleRegion] + debug visualization
  ↓
OCR Provider
  → list[OCRBlock]
  ↓
BubbleAssignmentService.assign()
  → list[ConversationRegion]
  ↓
OCRPostprocessService.process_page()
  (按 ConversationRegion 逐段修复)
  ↓
batch_pipeline (Translation)
  → 按 ConversationRegion 翻译而非 OCRBlock
  ↓
TextNormalizer
  ↓
TextLayoutService.layout(box=bubble_bbox)
  ↓
InpaintService.create_mask(regions=bubble_regions)
  (mask = 所有气泡内部区域，而非 OCR bbox)
  ↓
BackgroundCleaner
  ↓
TypographyService
  ↓
Output
```

---

## 文件变更清单

### 新增

| 文件 | 用途 |
|------|------|
| `models/bubble_region.py` | BubbleRegion / ConversationRegion 模型 |
| `services/bubble_detector.py` | OpenCV 气泡检测 |
| `services/bubble_assignment_service.py` | OCR → Bubble 映射 |
| `scripts/test_bubble_detection.py` | 气泡检测测试 |

### 修改

| 文件 | 改动 |
|------|------|
| `services/batch_prompt_builder.py` | `build_user_prompt()` 按 ConversationRegion 组织文本 |
| `services/text_layout_service.py` | `layout(box=bubble_bbox)` 使用气泡边界框 |
| `services/render_service.py` | 集成 BubblePipeline |
| `services/inpaint_service.py` | `create_mask(regions=bubble_regions)` 基于气泡区域 |
| `services/region_classifier.py` | 新增对 ConversationRegion 的分类 |
| `models/render_models.py` | 新增 BubbleRegion / ConversationRegion |

### 不修改

- `demo/batch_pipeline.py`（只改内部服务调用，不改入口）
- `providers/ocr/*`（OCR Provider 不修改）
- `services/ocr_postprocess_service.py`（不修改）

---

## 依赖分析

| 依赖 | 用于 | 状态 |
|------|------|------|
| opencv-python | 气泡检测（轮廓/形态学） | ✅ 已安装 |
| numpy | 图像处理 | ✅ 已安装 |
| PIL/Pillow | 图像操作 | ✅ 已安装 |
| scikit-image | 可选的增强型分割 | ❌ 不需要 |
| torch | BallonsTranslator 需要 | ⚠️ 已安装（EasyOCR）|
| paddleocr | BallonsTranslator 需要 | ❌ 不需要 |

**零新增依赖。**

---

## 实施计划

| Phase | 内容 | 文件数 | 依赖 |
|-------|------|--------|------|
| P1 | BubbleDetector + debug | 3 新增 | opencv ✅ |
| P2 | BubbleAssignmentService | 1 新增 | P1 |
| P3 | BubblePromptBuilder 修改 | 2 修改 | P2 |
| P4 | BubbleLayout 修改 | 2 修改 | P2 |
| P5 | RenderPipeline 集成 | 3 修改 | P1-P4 |

---

## 风险

| 风险 | 概率 | 缓解 |
|------|------|------|
| 白底气泡检测不准（非白色气泡） | 中 | 增加 Canny 边缘检测 + 形态学闭合备选 |
| 文字间隙过大，形态学无法闭合 | 低 | 增大 kernel 尺寸至 25x25 |
| 气泡尾部（三角指针）断开 | 低 | dilate 操作连接断开的轮廓 |
| 日本漫画右侧阅读气泡排序 | 低 | 排序方式后续处理，不在 bubble detection 中 |