from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import json
import os
import time
from pathlib import Path
from typing import TextIO

import torch
from diffusers import ComponentsManager, ModularPipeline
from diffusers.utils.export_utils import encode_video


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_NAME = Path(__file__).name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VBench prompts with the lossless MiniMax-H3 Base schedule on one or two H200 GPUs."
    )
    parser.add_argument("--prompts-csv", type=Path, default=REPO_ROOT / "VBench-origin_256_prompts.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "vbench_origin_256_hq")
    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / "models" / "MiniMax-H3")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Sigma grid points including terminal zero; 50 means 49 transformer evaluations.",
    )
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None, help="Exclusive end index; defaults to all prompts.")
    parser.add_argument(
        "--memory-reserve-margin",
        default="16GB",
        help="GPU memory kept free by Diffusers auto CPU offload on a single visible GPU.",
    )
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


def acquire_run_lock(output_dir: Path) -> TextIO | None:
    lock_path = output_dir / ".infer_vbench_256.lock"
    handle = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.write(f"pid={os.getpid()}\n")
    handle.flush()
    return handle


def build_pipelines(args: argparse.Namespace):
    model_path = str(args.model_path.resolve())
    workflow = ModularPipeline.from_pretrained(model_path, local_files_only=True).blocks.get_workflow("t2va")

    # MiniMax-H3 denoises video and audio jointly, but this benchmark requests a
    # silent MP4. Removing the audio decoder avoids loading the audio VAE and
    # guarantees that the output container has no audio stream.
    workflow.sub_blocks.pop("decode.audio")

    if torch.cuda.device_count() == 1:
        # A single manager coordinates Qwen and the 61.7 GB transformer so only
        # the component currently executing occupies the H200.
        generation_manager = ComponentsManager()
        generator_pipeline = workflow.init_pipeline(model_path, components_manager=generation_manager)
        generator_pipeline.load_components(
            dtype=torch.bfloat16,
            pretrained_model_name_or_path=model_path,
            local_files_only=True,
        )
        generation_manager.enable_auto_cpu_offload(
            device="cuda:0",
            memory_reserve_margin=args.memory_reserve_margin,
        )
        conditioner = None
    else:
        # With two H200s, Qwen and generation each fit in BF16 on their own card.
        text_workflow = workflow.sub_blocks.pop("text_encoder")
        text_manager = ComponentsManager()
        conditioner = text_workflow.init_pipeline(model_path, components_manager=text_manager)
        conditioner.load_components(
            dtype=torch.bfloat16,
            pretrained_model_name_or_path=model_path,
            local_files_only=True,
        )
        text_manager.enable_auto_cpu_offload(device="cuda:1")

        generation_manager = ComponentsManager()
        generator_pipeline = workflow.init_pipeline(model_path, components_manager=generation_manager)
        generator_pipeline.load_components(
            dtype=torch.bfloat16,
            pretrained_model_name_or_path=model_path,
            local_files_only=True,
        )
        generation_manager.enable_auto_cpu_offload(device="cuda:0")
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
    if args.fps != 24:
        raise ValueError("MiniMax-H3 generates at its canonical 24 FPS")
    if args.num_inference_steps < 2:
        raise ValueError("MiniMax-H3 needs at least two sigma grid points including terminal zero")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[{SCRIPT_NAME}] prompts={len(rows)} range=[{start}, {end}) "
        f"frames={args.num_frames} size={args.width}x{args.height} fps={args.fps} "
        f"sigma_points={args.num_inference_steps} nfe={args.num_inference_steps - 1} "
        f"precision=bf16 audio=off"
    )
    if args.dry_run:
        for index in range(start, min(end, start + 3)):
            print(f"video_{index:03d}.mp4 <- {rows[index]['prompt']}")
        return

    if torch.cuda.device_count() not in (1, 2):
        raise RuntimeError(
            f"Expected one or two visible GPUs, found {torch.cuda.device_count()}. "
            "Use one H200 for smoke tests or two H200s for the batch."
        )
    if not args.model_path.exists():
        raise FileNotFoundError(args.model_path)

    run_lock = acquire_run_lock(output_dir)
    if run_lock is None:
        print(f"[{SCRIPT_NAME}] another process is already generating into {output_dir}; exiting cleanly")
        return
    if not args.overwrite and all(is_complete(output_dir / f"video_{index:03d}.mp4") for index in range(start, end)):
        print(f"[{SCRIPT_NAME}] all requested videos already exist in {output_dir}; nothing to do")
        return

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
        generation_args = {
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "generator": torch.Generator(device="cpu").manual_seed(seed),
            "output": ["videos"],
        }
        state = None
        if conditioner is None:
            results = generator_pipeline(prompt=row["prompt"], **generation_args)
        else:
            state = conditioner(prompt=row["prompt"])
            results = generator_pipeline(state=state, **generation_args)
        temp_path = output_dir / f".video_{index:03d}.partial.mp4"
        temp_path.unlink(missing_ok=True)
        encode_video(results["videos"][0], fps=args.fps, output_path=str(temp_path))
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
                "quality_profile": "lossless-base",
                "sigma_points": args.num_inference_steps,
                "transformer_evaluations": args.num_inference_steps - 1,
                "video_flow_shift": 12.0,
                "audio_flow_shift": 3.0,
                "num_frames": args.num_frames,
                "width": args.width,
                "height": args.height,
                "fps": args.fps,
                "audio": False,
            },
        )
        print(f"[{index + 1:03d}/{len(rows)}] saved {output_path.name} in {elapsed:.1f}s", flush=True)
        del results
        if state is not None:
            del state
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
