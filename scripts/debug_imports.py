#!/usr/bin/env python
"""
调试导入路径：对比 script 模式和 -c 模式的 sys.path 差异

用法:
  uv run python -c "exec(open('scripts/debug_imports.py').read())"
  uv run python scripts/debug_imports.py
"""

import site
import sys

print("=" * 60)
print("ENVIRONMENT DEBUG")
print("=" * 60)
print(f"sys.executable:  {sys.executable}")
print(f"sys.prefix:      {sys.prefix}")
print(f"sys.base_prefix: {sys.base_prefix}")
print(f"VIRTUAL_ENV:     {__import__('os').environ.get('VIRTUAL_ENV', '(not set)')}")
print(f"PYTHONPATH:      {__import__('os').environ.get('PYTHONPATH', '(not set)')}")
print(f"PYTHONHOME:      {__import__('os').environ.get('PYTHONHOME', '(not set)')}")
print()

try:
    sp = site.getsitepackages()
    print(f"site.getsitepackages(): {sp}")
except Exception:
    pass
try:
    usp = site.getusersitepackages()
    print(f"site.getusersitepackages(): {usp}")
except Exception:
    pass
print()

print("sys.path (all):")
for i, p in enumerate(sys.path):
    print(f"  [{i}] {p}")
print()

# Check for import shadowing files
import pathlib

root = pathlib.Path(__file__).resolve().parent.parent
print("=" * 60)
print("IMPORT SHADOW CHECK")
print("=" * 60)
shadow_names = ["site.py", "typing.py", "platform.py", "paddle.py", "paddleocr.py"]
for name in shadow_names:
    for p in sys.path:
        candidate = pathlib.Path(p) / name
        if candidate.exists():
            print(f"  WARNING: {candidate} shadows import '{name.replace('.py', '')}'")

print()

# Try importing paddle from script mode
print("=" * 60)
print("IMPORT TEST (script mode)")
print("=" * 60)
for mod_name in ["paddle", "paddleocr"]:
    try:
        mod = __import__(mod_name)
        print(f"  {mod_name}: OK at {mod.__file__}")
    except ImportError as e:
        print(f"  {mod_name}: FAILED — {e}")

print()
print("To compare with -c mode, run:")
print("  uv run python -c \"exec(open('scripts/debug_imports.py').read())\"")