#!/usr/bin/env python3
"""
Worker entry point for outline generation only.
Called by Go worker with payload JSON, outputs outline JSON.
"""

import json
import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).resolve().parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from langchain_openai import ChatOpenAI
from models.assignment import AssignmentSpec, AssignmentStatus
from chains.assignment import build_assignment_outline_chain
from chains import build_shared_llm


def generate_outline_from_payload(payload_path: str, output_path: str):
    """Generate outline from JSON payload file."""
    print(f"[worker_generate_outline] Starting outline generation", flush=True)
    print(f"[worker_generate_outline] Payload: {payload_path}, Output: {output_path}", flush=True)
    
    with open(payload_path, "r") as f:
        payload = json.load(f)
    
    print(f"[worker_generate_outline] Loaded payload: {json.dumps(payload, indent=2)}", flush=True)

    # Initialize LLM
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    print(f"[worker_generate_outline] Initializing LLM...", flush=True)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    shared_llm = build_shared_llm(llm)
    print(f"[worker_generate_outline] LLM initialized", flush=True)

    # Build AssignmentSpec from payload
    spec = AssignmentSpec(
        assignment_type=payload.get("assignment_type"),
        description=payload.get("description"),
        length_words=payload.get("length_words"),
        length_pages=payload.get("length_pages"),
        format_style=payload.get("format_style"),
        course_name=payload.get("course_name"),
        allow_web_search=payload.get("allow_web_search", True),
        status=AssignmentStatus.READY,
    )
    if payload.get("humanizer_config"):
        hc = payload["humanizer_config"]
        spec.humanizer_config.enabled = hc.get("enabled", True)
        spec.humanizer_config.intensity = hc.get("intensity", "light")

    # Generate outline
    print(f"[worker_generate_outline] Building outline chain...", flush=True)
    outline_chain = build_assignment_outline_chain(shared_llm)
    print(f"[worker_generate_outline] Generating outline...", flush=True)
    outline = outline_chain.invoke(spec)
    print(f"[worker_generate_outline] Outline generated", flush=True)

    # Write outline to output file as JSON
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(outline, f, indent=2)
    
    print(f"[worker_generate_outline] Successfully generated outline: {output_path}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: worker_generate_outline.py <payload.json> <output.json>", file=sys.stderr, flush=True)
        sys.exit(1)
    try:
        generate_outline_from_payload(sys.argv[1], sys.argv[2])
    except Exception as e:
        print(f"[worker_generate_outline] ERROR: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
