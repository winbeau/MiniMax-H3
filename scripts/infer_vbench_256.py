from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import time
from pathlib import Path

import torch
from diffusers import ComponentsManager, ModularPipeline
from diffusers.utils.export_utils import encode_video


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_NAME = Path(__file__).name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all 256 VBench prompts with MiniMax-H3 split over two H200 GPUs."
    )
    parser.add_argument("--prompts-csv", type=Path, default=REPO_ROOT / "VBench-origin_256_prompts.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "vbench_origin_256")
    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / "models" / "MiniMax-H3")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None, help="Exclusive end index; defaults to all prompts.")
    parser.add_argument("--group-offload", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "prompt" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a 'prompt' column")
        rows = list(reader)
    if len(rows) != 256:
        raise ValueError(f"Expected 256 prompts in {path}, found {len(rows)}")
    if any(not row["prompt"].strip() for row in rows):
        raise ValueError(f"{path} contains an empty prompt")
    return rows


def is_complete(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 1024


def append_manifest(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_pipelines(args: argparse.Namespace):
    model_path = str(args.model_path.resolve())
    workflow = ModularPipeline.from_pretrained(model_path, local_files_only=True).blocks.get_workflow("t2va")

    # MiniMax-H3 denoises video and audio jointly, but this benchmark requests a
    # silent MP4. Removing the audio decoder avoids loading the audio VAE and
    # guarantees that the output container has no audio stream.
    workflow.sub_blocks["decode"].sub_blocks.pop("audio")

    text_workflow = workflow.sub_blocks.pop("text_encoder")
    text_manager = ComponentsManager()
    text_manager.enable_auto_cpu_offload(device="cuda:1")
    conditioner = text_workflow.init_pipeline(model_path, components_manager=text_manager)
    conditioner.load_components(
        dtype=torch.bfloat16,
        pretrained_model_name_or_path=model_path,
        local_files_only=True,
    )

    generation_manager = ComponentsManager()
    generation_manager.enable_auto_cpu_offload(device="cuda:0")
    generator_pipeline = workflow.init_pipeline(model_path, components_manager=generation_manager)
    generator_pipeline.load_components(
        dtype=torch.bfloat16,
        pretrained_model_name_or_path=model_path,
        local_files_only=True,
    )
    if args.group_offload:
        generator_pipeline.vae.to(dtype=torch.float16)
        generator_pipeline.transformer.enable_group_offload(
            onload_device=torch.device("cuda:0"),
            offload_device=torch.device("cpu"),
            offload_type="block_level",
            num_blocks_per_group=1,
        )
        generator_pipeline.vae.enable_group_offload(
            onload_device=torch.device("cuda:0"),
            offload_device=torch.device("cpu"),
            offload_type="leaf_level",
        )
    return conditioner, generator_pipeline


def main() -> None:
    args = parse_args()
    rows = load_rows(args.prompts_csv.resolve())
    start = max(0, args.start_index)
    end = len(rows) if args.end_index is None else min(len(rows), args.end_index)
    if start >= end:
        raise ValueError(f"Empty index range [{start}, {end})")
    if args.height % 32 or args.width % 32:
        raise ValueError("MiniMax-H3 height and width must be divisible by 32")
    if (args.num_frames - 5) % 17:
        raise ValueError("MiniMax-H3 num_frames must satisfy num_frames = 17 * k + 5")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[{SCRIPT_NAME}] prompts={len(rows)} range=[{start}, {end}) "
        f"frames={args.num_frames} size={args.width}x{args.height} fps={args.fps} audio=off"
    )
    if args.dry_run:
        for index in range(start, min(end, start + 3)):
            print(f"video_{index:03d}.mp4 <- {rows[index]['prompt']}")
        return

    if torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"Expected exactly two visible GPUs, found {torch.cuda.device_count()}. "
            "Launch with CUDA_VISIBLE_DEVICES=0,1."
        )
    if not args.model_path.exists():
        raise FileNotFoundError(args.model_path)

    conditioner, generator_pipeline = build_pipelines(args)
    manifest_path = output_dir / "manifest.jsonl"
    for index in range(start, end):
        row = rows[index]
        output_path = output_dir / f"video_{index:03d}.mp4"
        if is_complete(output_path) and not args.overwrite:
            print(f"[{index + 1:03d}/{len(rows)}] skip {output_path.name}")
            continue

        seed = args.seed_base + index
        print(f"[{index + 1:03d}/{len(rows)}] seed={seed} prompt={row['prompt']!r}", flush=True)
        started = time.monotonic()
        state = conditioner(prompt=row["prompt"])
        results = generator_pipeline(
            state=state,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            generator=torch.Generator().manual_seed(seed),
            output="videos",
        )
        temp_path = output_dir / f".video_{index:03d}.partial.mp4"
        temp_path.unlink(missing_ok=True)
        encode_video(results[0], fps=args.fps, output_path=str(temp_path))
        if not is_complete(temp_path):
            raise RuntimeError(f"MiniMax-H3 did not create a valid file: {temp_path}")
        temp_path.replace(output_path)
        elapsed = time.monotonic() - started
        append_manifest(
            manifest_path,
            {
                "index": index,
                "record_id": row.get("record_id"),
                "prompt": row["prompt"],
                "output": output_path.name,
                "seed": seed,
                "elapsed_seconds": round(elapsed, 3),
                "num_frames": args.num_frames,
                "width": args.width,
                "height": args.height,
                "fps": args.fps,
                "audio": False,
            },
        )
        print(f"[{index + 1:03d}/{len(rows)}] saved {output_path.name} in {elapsed:.1f}s", flush=True)
        del state, results
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
