#!/usr/bin/env python
"""
Comic Bubble YOLO 训练脚本

用法:
  python scripts/train_bubble_model.py                              # 默认训练
  python scripts/train_bubble_model.py --model yolo11n-seg.pt       # 用 nano
  python scripts/train_bubble_model.py --epochs 50 --batch 2        # 快速验证
  python scripts/train_bubble_model.py --resume                     # 继续训练

标注流程:
  1. pip install labelme
  2. labelme dataset/bubble/images/train/    # 标注气泡
  3. labelme dataset/bubble/images/val/      # 标注验证集
  4. python scripts/convert_labels.py --input dataset/bubble/images --output dataset/bubble/labels
  5. python scripts/train_bubble_model.py

标签:
  speech    → 0   # 对白气泡
  narration → 1   # 旁白框
  title     → 2   # 标题
  sfx       → 3   # 拟声词
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loguru import logger
from ultralytics import YOLO

logger.remove()
logger.add(sys.stderr, level="INFO")


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11-seg for comic bubble detection")
    parser.add_argument("--model", default="models/bubble/yolo11n-seg-bubble.pt",
                        help="基础模型 .pt 路径或 YOLO 预训练模型名（如 yolo11n-seg.pt）")
    parser.add_argument("--data", default="dataset/mixed_bubble/dataset.yaml", help="数据集配置")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=4, help="batch size (RTX 4060 6GB: 4 for 1600px)")
    parser.add_argument("--imgsz", type=int, default=1600, help="输入图片尺寸")
    parser.add_argument("--device", default="0", help="训练设备: 0=cuda, cpu")
    parser.add_argument("--resume", action="store_true", help="从 checkpoint 继续训练")
    parser.add_argument("--patience", type=int, default=20, help="early stopping 轮数")
    parser.add_argument("--name", default="bubble", help="实验名称 (runs/segment/{name}/)")
    args = parser.parse_args()

    logger.info(f"模型: {args.model}")
    logger.info(f"数据: {args.data}")
    logger.info(f"Epochs: {args.epochs}, Batch: {args.batch}, Imgsz: {args.imgsz}")

    # 验证数据集
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"数据集配置文件不存在: {args.data}")
        sys.exit(1)

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        patience=args.patience,
        resume=args.resume,
        name=args.name,
        seed=42,
        amp=True,
        workers=2,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        cos_lr=True,
        overlap_mask=True,
        mask_ratio=4,
        val=True,
        save=True,
        save_period=10,
    )

    # 验证
    logger.info("=" * 50)
    logger.info("验证结果:")
    val_results = model.val()
    logger.info(f"  mAP50:     {val_results.box.map50:.4f}")
    logger.info(f"  mAP50-95:  {val_results.box.map:.4f}")

    # 导出模型（用 basename + 实验名，避免 model 带路径时产生嵌套路径）
    output_path = Path(f"models/bubble/{Path(args.model).stem}-{args.name}.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_pt = Path(f"runs/segment/{args.name}/weights/best.pt")
    if best_pt.exists():
        import shutil
        shutil.copy(str(best_pt), str(output_path))
        logger.info(f"模型已保存: {output_path}")
        logger.info(f"使用: YOLO11BubbleProvider('{output_path}')")
    else:
        logger.warning(f"未找到 best.pt: {best_pt}")


if __name__ == "__main__":
    main()