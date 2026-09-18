#!/usr/bin/env python3
"""List models available to a SleepyAI key without exposing the key."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.providers.sleepyai_catalog import discover_models


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="https://www.sleepyai.org/api/v1")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        print("ERROR: set LLM_API_KEY in the environment; never pass it on the command line.")
        return 2
    try:
        models = await discover_models(api_key, args.base_url)
    except Exception as exc:  # noqa: BLE001 - command reports a sanitized error category
        print(f"ERROR: model discovery failed ({type(exc).__name__})")
        return 1

    print(f"Discovered {len(models)} accessible models")
    for model in models:
        price_in = model.get("inputPrice", "?")
        price_out = model.get("outputPrice", "?")
        context = model.get("contextWindow", model.get("context_window", "?"))
        print(f"- {model['id']}  context={context} input={price_in} output={price_out}")

    if args.report:
        payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "base_url": args.base_url,
            "models": models,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Report written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
