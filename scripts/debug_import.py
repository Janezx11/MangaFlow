# scripts/debug_import.py

import sys

print(sys.executable)

try:
    import paddle
    print("PADDLE OK")
    print(paddle.__file__)
except Exception as e:
    print("PADDLE FAIL")
    print(repr(e))

try:
    import paddleocr
    print("PADDLEOCR OK")
    print(paddleocr.__file__)
except Exception as e:
    print("PADDLEOCR FAIL")
    print(repr(e))
    