# OCR Provider Compatibility Matrix

## Tested Configurations

| PaddleOCR | PaddlePaddle | Python | OS | Status |
|-----------|-------------|--------|-----|--------|
| 2.10.0 | 3.3.1 | 3.13 | Windows | ⚠️ Inference may fail (OneDnnContext error) |
| 2.8.1 | 2.6.1 | 3.10 | Windows | ✅ Works |
| 2.8.1 | 2.6.1 | 3.10 | Linux | ✅ Works |

## Common Issues

### 1. `OneDnnContext does not have the input Filter` (PaddleOCR 2.10.0 + Paddle 3.3.1)

**Symptom**: PaddleOCR initializes OK but crashes on `ocr.ocr()` call.

**Workarounds**:
- Set `use_gpu: false` and `enable_mkldnn: false` in config
- Downgrade to paddlepaddle 2.6.1 + paddleocr 2.8.1
- Switch to EasyOCR fallback (auto-detected by DI container)

### 2. `ImportError: No module named 'paddle'`

**Fix**: `uv sync --extra paddle`

### 3. Python 3.13 compatibility

PaddlePaddle 3.x has experimental Python 3.13 support. If issues persist:
- Use Python 3.10–3.12
- Or use EasyOCR (better Python 3.13 support)

## Fallback Chain

When PaddleOCR fails, the system automatically tries:

1. PaddleOCR → 2. EasyOCR → 3. Empty result (graceful degradation)

No OCR failure will crash the translation pipeline.

## Recommended Setup

```yaml
ocr:
  provider: paddle_ocr
  providers:
    paddle_ocr:
      lang: en
      use_gpu: false      # default: false (avoid GPU issues)
      enable_mkldnn: false # default: false (avoid OneDnnContext crash)
    easyocr:
      lang: en
      use_gpu: false
  fallback:
    - easyocr
```