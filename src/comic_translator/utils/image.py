from pathlib import Path

from PIL import Image


def load_image(path: str | Path) -> Image.Image:
    """加载图片文件"""
    return Image.open(path)


def save_image(image: Image.Image, path: str | Path) -> None:
    """保存图片文件"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def get_image_size(path: str | Path) -> tuple[int, int]:
    """获取图片尺寸"""
    with Image.open(path) as img:
        return img.size
