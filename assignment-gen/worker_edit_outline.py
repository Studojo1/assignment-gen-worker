#!/usr/bin/env python3
"""
Worker entry point for outline editing.
Called by Go worker with outline JSON and user message, outputs updated outline JSON.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).resolve().parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from chains.azure_openai import build_azure_chat_llm, require_azure_openai_env
from chains.assignment import build_assignment_outline_editor_chain
from chains import build_shared_llm


async def edit_outline_from_payload(payload_path: str, output_path: str):
    """Edit outline from JSON payload file."""
    print(f"[worker_edit_outline] Starting outline editing", flush=True)
    print(f"[worker_edit_outline] Payload: {payload_path}, Output: {output_path}", flush=True)
    
    with open(payload_path, "r") as f:
        payload = json.load(f)
    
    print(f"[worker_edit_outline] Loaded payload: {json.dumps(payload, indent=2)}", flush=True)

    outline = payload.get("outline", {})
    user_message = payload.get("user_message", "")
    
    if not outline or not user_message:
        raise ValueError("outline and user_message are required in payload")

    # Initialize LLM
    require_azure_openai_env()

    print(f"[worker_edit_outline] Initializing LLM...", flush=True)
    llm = build_azure_chat_llm(temperature=0.1)
    shared_llm = build_shared_llm(llm)
    print(f"[worker_edit_outline] LLM initialized", flush=True)

    # Edit outline
    print(f"[worker_edit_outline] Building outline editor chain...", flush=True)
    editor_chain = build_assignment_outline_editor_chain(shared_llm)
    print(f"[worker_edit_outline] Editing outline...", flush=True)
    result = await editor_chain.ainvoke({
        "outline": outline,
        "user_message": user_message,
    })
    print(f"[worker_edit_outline] Outline edited", flush=True)

    # Write result to output file as JSON
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    output_data = {
        "outline": result.outline,
        "assistant_message": result.assistant_message,
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"[worker_edit_outline] Successfully edited outline: {output_path}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: worker_edit_outline.py <payload.json> <output.json>", file=sys.stderr, flush=True)
        sys.exit(1)
    try:
        asyncio.run(edit_outline_from_payload(sys.argv[1], sys.argv[2]))
    except Exception as e:
        print(f"[worker_edit_outline] ERROR: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
