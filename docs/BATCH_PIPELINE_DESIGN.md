# Batch Translation Pipeline — 重构设计

## 第一部分：当前架构问题分析

### 问题 1：逐页 API 调用

`demo/batch_translate.py` 第 175 行的 `for page_idx, img_path in enumerate(images)` 循环内，每页调用一次 `translation_service.translate_page()` → 每次调 `QwenTranslator.translate()` → 每次发一个 `/api/generate` 请求。

100 页漫画 = 100 次 API 调用。迁移到 SenseNova 后按调用次数计费，成本翻 100 倍。

### 问题 2：逐页上下文隔离

每次请求只传当前页的 OCR 文本。模型看不到前后页的对话，导致：
- 角色名前后不一致（上页 "Tony Stark"，下页 "托尼·斯塔克" 可能对不上）
- 剧情断裂（第 5 页的 "他" 模型不知道指的是第 3 页出现的角色）

目前的 `PageSummaryService` 尝试缓解此问题，但摘要丢失了大量细节。

### 问题 3：QwenTranslator 一次性输出

`QwenTranslator._parse_response()` 期望模型返回完整 JSON 数组。当模型输出被截断或格式错误时，整个 Batch 的翻译都丢失。只有 fallback 到原文一条路。

### 问题 4：OCR 重复扫描

`batch_translate.py` 的循环内 OCR + Normalize + Translate 串行。断点续跑时如果翻译缓存命中但 OCR 缓存过期，仍需要重做 OCR。

### 问题 5：缓存粒度不匹配

当前缓存粒度是 `page` 级别。Batch 模式下需要 `batch` 级别缓存。

### 可以复用的模块

| 模块 | 状态 | 重用方式 |
|------|------|---------|
| `OCRNormalizeService` | ✅ 可复用 | 先 OCR 全部页面，统一 Normalize |
| `GlossaryService` | ✅ 可复用 | Batch 1 提取 Glossary 后写入 SQLite，后续 Batch 读取 |
| `CharacterIdentityService` | ✅ 可复用 | 角色解析逻辑不变 |
| `CacheService` / `FileCacheProvider` | ⚠️ 需扩展 | 新增 Batch 级缓存 |
| `PaddleOCRProvider` / `EasyOCRProvider` | ✅ 可复用 | 不变 |
| `LLMClient` / `OllamaClient` | ✅ 可复用 | 未来换 SenseNovaClient |
| `TranslationContext` | ⚠️ 需改造 | 需要容纳多页数据 |

### 可以删除的模块

| 模块 | 理由 |
|------|------|
| `PageSummaryService` | Batch 模式下不再需要逐页摘要，模型直接看到前后文 |
| `PromptCacheService` | Batch 模式下 Prompt 太大，缓存意义小 |
| `SemanticTranslationMemoryService` | Batch 模式下 Glossary 取代 TM 的单句匹配 |
| `ContextAssembler` | 被新的 BatchContextAssembler 取代 |
| `TranslationService` | 被新的 BatchTranslationService 取代 |

---

## 第二部分：新架构设计

### 核心流程

```
Step 1: OCR All Pages
  [img_001.png, ..., img_100.png]
  → [page_1_blocks, ..., page_100_blocks]
  → data/cache/{project}/ocr/all_pages_ocr.json

Step 2: Split into Batches
  Batch 1: pages 1-20
  Batch 2: pages 21-40
  ...

Step 3: Process Batch 1
  → build_prompt(pages_1_20_blocks)
  → API call (1 次)
  → parse response → extract glossary → save translations
  → data/cache/{project}/batch/batch_001.json
  → glossary entries → SQLite

Step 4: Process Batch 2 (with glossary from Batch 1)
  → build_prompt(pages_21_40_blocks + glossary)
  → API call (1 次)
  → parse response → save translations
  ...

Step 5: Assemble Results
  → merge all batch outputs
  → write final JSON
```

### 新增文件

| 文件 | 用途 |
|------|------|
| `services/batch_ocr_service.py` | 先 OCR 全部页面，统一输出 |
| `services/batch_translation_service.py` | Batch 翻译编排 |
| `services/batch_prompt_builder.py` | 构建多页 Prompt |
| `services/batch_cache.py` | Batch 级别缓存 |
| `models/batch_result.py` | BatchResult 数据模型 |
| `providers/translation/sensenova_translator.py` | SenseNova 客户端 |
| `demo/batch_pipeline.py` | 新的入口脚本 |

### 新增数据模型

```python
class BatchPage(BaseModel):
    page: int
    blocks: list[NormalizedOCRBlock]
    metadata: dict = {}

class BatchInput(BaseModel):
    pages: list[BatchPage]
    glossary: list[GlossaryEntry] = []
    batch_index: int = 0

class BatchResult(BaseModel):
    batch_index: int
    pages: list[PageTranslation]
    glossary_extracted: dict[str, str]
    raw_response: str = ""

class PageTranslation(BaseModel):
    page: int
    blocks: list[TranslationBlock]
```

### 新增数据库表

`TranslatedPage` 表，用于存储翻译结果，支持断点续跑：

```sql
CREATE TABLE translated_pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    batch_index INTEGER NOT NULL,
    page INTEGER NOT NULL,
    page_data TEXT NOT NULL,  -- JSON
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP,
    UNIQUE(project_id, page)
);
```

---

## 第三部分：Prompt 设计

### System Prompt

```
You are a professional comic translator with expertise in OCR error correction 
and English-to-Chinese translation.

Your tasks:
1. Correct OCR errors in the source text
2. Translate the corrected text to Simplified Chinese
3. Maintain consistency of character names, locations, and titles
4. Preserve speaking style and emotional tone

Output ONLY valid JSON. No explanations, no comments.
```

### User Prompt (Batch 1 — 无 Glossary)

```
Project: "{project_name}"
Comic Style: {comic_style}
Language: English → Simplified Chinese

Glossary: (empty — this is the first batch)

Pages to translate:
=== PAGE 1 ===
[id=p1_b1] Hello Tony.
[id=p1_b2] We have a problem.

=== PAGE 2 ===
[id=p2_b1] Mr. Stark, the reactor is unstable.

---

Output JSON format:
{
  "glossary": {
    "Tony Stark": "托尼·斯塔克",
    "Stark Tower": "斯塔克大厦"
  },
  "pages": [
    {
      "page": 1,
      "blocks": [
        {"id": "p1_b1", "corrected": "Hello Tony.", "translated": "你好，托尼。"},
        {"id": "p1_b2", "corrected": "We have a problem.", "translated": "我们出问题了。"}
      ]
    },
    {
      "page": 2,
      "blocks": [
        {"id": "p2_b1", "corrected": "Mr. Stark, the reactor is unstable.", "translated": "斯塔克先生，反应堆不稳定。"}
      ]
    }
  ]
}
```

### User Prompt (Batch 2+ — 带 Glossary)

```
...同上...

Glossary (from previous batches):
  Tony Stark → 托尼·斯塔克
  Stark Tower → 斯塔克大厦
  Iron Man → 钢铁侠

Use the above glossary for consistent translation.
Do NOT retranslate these names differently.

Pages to translate:
...
```

---

## 第四部分：改造步骤

### Step 1: 新增数据模型

新建 `models/batch_result.py`：

```python
class BatchPage(BaseModel):
    page: int
    blocks: list[NormalizedOCRBlock]

class PageTranslation(BaseModel):
    page: int
    blocks: list[TranslationBlock]

class BatchInput(BaseModel):
    batch_index: int
    pages: list[BatchPage]
    glossary: list[GlossaryEntry] = []

class BatchOutput(BaseModel):
    batch_index: int
    pages: list[PageTranslation]
    glossary: dict[str, str] = {}
```

### Step 2: 新增 BatchCache

新建 `services/batch_cache.py`：

```python
class BatchCacheService:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)

    def get_ocr_cache(self, project_id: str) -> list[BatchPage] | None:
        """检查 all_pages_ocr.json 是否存在"""
    
    def set_ocr_cache(self, project_id: str, pages: list[BatchPage]):
    
    def get_batch_output(self, project_id: str, batch_index: int) -> BatchOutput | None:
        """检查 batch_{idx}.json 是否存在"""
    
    def set_batch_output(self, project_id: str, batch_index: int, output: BatchOutput):
```

### Step 3: 新增 BatchOCRService

新建 `services/batch_ocr_service.py`：

```python
class BatchOCRService:
    """
    1. 遍历 data/comics/ 下所有图片
    2. 调用 OCRProvider.detect_text()
    3. 调用 OCRNormalizeService.process()
    4. 输出 list[BatchPage]
    5. 缓存到 all_pages_ocr.json
    """
```

### Step 4: 新增 BatchPromptBuilder

新建 `services/batch_prompt_builder.py`：

```python
class BatchPromptBuilder:
    def build_system_prompt(self) -> str:
    
    def build_user_prompt(
        self,
        pages: list[BatchPage],
        glossary: list[GlossaryEntry],
        batch_index: int,
        project_name: str = "",
        comic_style: str = "western",
    ) -> str:
        """
        组装多页 + Glossary 的 User Prompt
        """
    
    def parse_response(self, response: str) -> BatchOutput:
        """
        解析模型返回的 JSON
        包含: glossary + pages[].blocks[].corrected + translated
        """
```

### Step 5: 新增 BatchTranslationService

新建 `services/batch_translation_service.py`：

```python
class BatchTranslationService:
    """
    1. 从 BatchOCRService 获取所有页的 OCR 结果
    2. 按 batch_size 切分
    3. 逐 Batch:
       a. 检查 BatchCache 命中 → 跳过
       b. 从 GlossaryService 读取已有 Glossary
       c. BatchPromptBuilder.build_user_prompt()
       d. LLMClient.generate()
       e. BatchPromptBuilder.parse_response()
       f. 提取 Glossary 写入 GlossaryService
       g. 翻译结果写入 TranslatedPage 表
       h. 写入 BatchCache
    4. 全部完成后合并输出
    """
```

### Step 6: 新增 demo/batch_pipeline.py

新的入口脚本，替代 `demo/batch_translate.py`：

```python
async def run_pipeline(
    project_name: str,
    batch_size: int = 20,
    source_lang: str = "en",
    target_lang: str = "zh",
    comic_style: str = "western",
):
    """
    完整流程：
    1. OCR 全部页面
    2. 按 batch_size 切分
    3. 逐 Batch 调 API
    4. 自动提取 Glossary
    5. 输出最终结果
    """
```

### Step 7: 迁移现有数据

`CharacterMemoryService` 和 `GlossaryService` 的现有数据无需迁移，BatchPipeline 直接复用同一 SQLite 数据库。

---

## 第五部分：性能分析

### 假设

- 100 页漫画
- 每页 20 个文本框（OCR 后）
- 每个文本框平均 10 个 token（约 40 字符）
- API 按调用次数计费（SenseNova）

### 方案对比

| 指标 | 方案 A: 逐页 | 方案 B: 20页/Batch | 方案 C: 50页/Batch |
|------|-------------|-------------------|-------------------|
| API 调用次数 | 100 | **5** | **2** |
| 每次 Prompt 大小 | ~4K tokens | ~80K tokens | ~200K tokens |
| 总 Token 消耗(含复用) | ~400K + 系统性开销 | ~400K + 少量系统性开销 | ~400K + 系统性开销 |
| 调用成本(SenseNova) | **100x** | **5x** | **2x** |
| 角色一致性 | 差(靠 TM/Summary) | 好(同 Batch 内可见) | 最好(全量可见) |
| 剧情连续性 | 差 | 好 | 最好 |
| 模型上下文要求 | 低(4K) | 中(128K 足够) | 中高(128K 足够) |
| 单次请求失败影响 | 1 页 | 20 页 | 50 页 |
| 首次 Token 加载 | 100 次 | 5 次 | 2 次 |
| OCR 阶段耗时 | 串行 | 先并行扫完 | 先并行扫完 |

### 推荐方案：**方案 B（20页/Batch）**

理由：
1. **API 调用次数：100 → 5**，成本降至 1/20
2. **20 页的 Prompt 约 80K tokens**，qwen3.5:9b 的 262K 上下文完全够用
3. **单 Batch 失败损失 20 页**而非 50 页，可接受
4. **20 页内角色通常不变**，一致性足够
5. **Glossary 跨 Batch 传递**，即使角色跨 Batch 也能保持一致

### 成本估算（SenseNova）

```
逐页方案: 100 calls
Batch 方案: 5 calls

成本降低: 95%
```

如果 SenseNova 每次调用 $0.01，则：
- 逐页: $1.00/章
- Batch: $0.05/章
- 100 章漫画: $100 vs $5
