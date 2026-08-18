#!/usr/bin/env python
"""
Ollama Client 连接测试工具

测试:
  1. 连接 — Ollama 是否在运行
  2. 模型列表 — 可用模型
  3. 基础生成 — 调用 qwen3.5:9b 确认可工作

用法:
  uv run python scripts/test_ollama_client.py
  uv run python scripts/test_ollama_client.py --model llama3.2:3b
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


async def main():
    parser = argparse.ArgumentParser(description="Ollama Client 连接测试")
    parser.add_argument("--model", default="qwen3.5:9b", help="模型名称")
    parser.add_argument("--url", default="http://localhost:11434", help="Ollama 地址")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细输出")
    args = parser.parse_args()

    from comic_translator.clients.ollama_client import OllamaClient

    client = OllamaClient(model=args.model, base_url=args.url, timeout=300)

    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    print("=" * 60)
    print("Ollama Client Test")
    print("=" * 60)
    print(f"  URL:   {args.url}")
    print(f"  Model: {args.model}")
    print()

    # Step 1: 连接测试
    print("-" * 60)
    print("Step 1: Connection Test")
    print("-" * 60)
    try:
        ok = await client.is_available()
        print(f"  Connection: {'OK' if ok else 'FAILED'}")
        if not ok:
            print("  Is Ollama running? Try: ollama serve")
            sys.exit(1)
    except Exception as e:
        print(f"  Connection: FAILED — {e}")
        print("  Is Ollama running? Try: ollama serve")
        sys.exit(1)

    # Step 2: 模型列表
    print()
    print("-" * 60)
    print("Step 2: Model List")
    print("-" * 60)
    try:
        models = await client.list_models()
        if models:
            print(f"  Available models ({len(models)}):")
            for m in models:
                name = m.get("name", m.get("model", "?"))
                size = m.get("size", 0)
                size_gb = size / 1e9
                print(f"    - {name} ({size_gb:.1f} GB)")
        else:
            print("  No models found. Run: ollama pull qwen3.5:9b")
    except Exception as e:
        print(f"  FAILED — {e}")

    # Step 3: 基础生成
    print()
    print("-" * 60)
    print(f"Step 3: Generate Test (model={args.model})")
    print("-" * 60)
    try:
        response = await client.generate(
            prompt="Say 'Hello, Ollama is working!' in 5 words or less.",
            temperature=0.0,
            max_tokens=50,
        )
        safe = response.strip().encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        print(f"  Response: {safe}")
        print(f"  Length:   {len(response.strip())} chars")
        print(f"  Status: OK")
    except Exception as e:
        import traceback
        print(f"  FAILED — {type(e).__name__}: {e}")
        print(f"  Traceback: {traceback.format_exc()[:300]}")
        print(f"\n  Common causes:")
        print(f"  1. Model '{args.model}' not pulled: ollama pull {args.model}")
        print(f"  2. First load is slow (6.6GB model) — try again, it's cached after first use")
        print(f"  3. Not enough VRAM/RAM")
        sys.exit(1)

    # Step 4: Token 估算
    print()
    print("-" * 60)
    print("Step 4: Token Count (local estimate)")
    print("-" * 60)
    test_texts = [
        "Hello, world!",
        "你好，世界！",
        "Tony Stark arrives at Stark Tower. J.A.R.V.I.S., status report?",
    ]
    for text in test_texts:
        tokens = await client.count_tokens(text)
        print(f"  {str(text)[:50]:50s} ~ {tokens} tokens")

    # Summary
    print()
    print("=" * 60)
    print("All checks passed — OllamaClient is working.")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())