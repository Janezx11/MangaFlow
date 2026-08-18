# Examples & Demo Scripts

## 快速开始

```bash
# 初始化项目
python -m comic_translator init my_project --input data/comics

# 运行翻译流水线
python -m comic_translator run --project <project-id>

# 启动 GUI
python -m comic_translator.gui.app
```

## 独立示例脚本

| 脚本 | 说明 |
|------|------|
| `batch_translate.py` | 完整翻译流水线（OCR → 翻译 → 保存） |
| `render_pipeline.py` | 渲染流水线（OCR → 排版 → 输出） |
| `text_detection_pipeline.py` | 文字检测 + 气泡检测可视化 |
| `check_ocr.py` | 快速测试 OCR 效果 |
| `quality_report.py` | 生成质检报告 |

## 模型训练脚本

参见 `scripts/` 目录：
- `train_bubble_model.py` — 训练气泡检测模型
- `evaluate_bubble_model.py` — 评估模型
- `merge_bubble_datasets.py` — 合并数据集
- `benchmark_detection.py` — 检测模型基准测试