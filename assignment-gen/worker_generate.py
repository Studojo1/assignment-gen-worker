#!/usr/bin/env python3
"""
Worker entry point for assignment generation.
Called by Go worker with payload JSON and output path.
"""

import asyncio
import json
import os
import sys
import logging
from pathlib import Path

# Configure logging to output to stdout with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
    force=True  # Override any existing config
)

# Add src to path
src_path = Path(__file__).resolve().parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from chains.azure_openai import build_azure_chat_llm, require_azure_openai_env
from models.assignment import AssignmentSpec, AssignmentStatus
from chains.assignment import (
    build_assignment_outline_chain,
    build_assignment_pipeline,
)
from chains import build_shared_llm


async def generate_from_payload(payload_path: str, output_path: str):
    """Generate assignment from JSON payload file."""
    print(f"[worker_generate] Starting assignment generation", flush=True)
    print(f"[worker_generate] Payload: {payload_path}, Output: {output_path}", flush=True)
    
    with open(payload_path, "r") as f:
        payload = json.load(f)
    
    print(f"[worker_generate] Loaded payload: {json.dumps(payload, indent=2)}", flush=True)

    # Initialize LLM
    require_azure_openai_env()

    print(f"[worker_generate] Initializing LLM...", flush=True)
    llm = build_azure_chat_llm(temperature=0.1)
    shared_llm = build_shared_llm(llm)
    print(f"[worker_generate] LLM initialized", flush=True)

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

    # Use provided outline or generate new one
    outline = None
    if "outline" in payload and payload["outline"]:
        print(f"[worker_generate] Using provided outline from payload", flush=True)
        outline = payload["outline"]
    else:
        # Generate outline
        print(f"[worker_generate] Building outline chain...", flush=True)
        outline_chain = build_assignment_outline_chain(shared_llm)
        print(f"[worker_generate] Generating outline...", flush=True)
        outline = outline_chain.invoke(spec)
        print(f"[worker_generate] Outline generated", flush=True)

    # Generate assignment
    print(f"[worker_generate] Building assignment pipeline...", flush=True)
    pipeline = build_assignment_pipeline(shared_llm)
    
    # Log outline sections for progress tracking
    if outline and "sections" in outline:
        num_sections = len(outline["sections"])
        print(f"[worker_generate] Starting generation for {num_sections} sections", flush=True)
        for i, section in enumerate(outline["sections"], 1):
            section_id = section.get("id", "?")
            section_title = section.get("title", "Untitled")
            print(f"[worker_generate] Section {i}/{num_sections} (ID: {section_id}): {section_title}", flush=True)
    
    print(f"[worker_generate] Running pipeline...", flush=True)
    result = await pipeline(
        spec,
        outline=outline,
        courseplan_analysis=spec.description or "",
        user_responses="",
        corpus_texts=None,
        output_filename=os.path.basename(output_path),
    )
    print(f"[worker_generate] Pipeline completed, result: {result.doc_path}", flush=True)

    # Move generated file to output path
    import shutil
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    print(f"[worker_generate] Moving file from {result.doc_path} to {output_path}", flush=True)
    shutil.move(result.doc_path, output_path)

    print(f"[worker_generate] Successfully generated: {output_path}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: worker_generate.py <payload.json> <output.docx>", file=sys.stderr, flush=True)
        sys.exit(1)
    try:
        asyncio.run(generate_from_payload(sys.argv[1], sys.argv[2]))
    except Exception as e:
        print(f"[worker_generate] ERROR: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
