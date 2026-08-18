from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_gpu=False,
    lang="en",
    show_log=True,
)

result = ocr.ocr(
    "data/comics/0002_GREECE0001.png"
)

print(result)python demo/batch_translate.py