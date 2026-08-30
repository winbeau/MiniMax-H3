from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import ComponentsManager, ModularPipeline
from diffusers.utils.export_utils import encode_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MiniMax-H3 two-H200 T2VA smoke inference")
    parser.add_argument("--model-path", default="models/MiniMax-H3")
    parser.add_argument("--output", default="outputs/minimax-h3-t2va-smoke.mp4")
    parser.add_argument(
        "--prompt",
        default=(
            "A red fox trots through a snowy pine forest while the camera tracks alongside it. "
            "Snow crunches under its paws and a soft winter wind moves the branches."
        ),
    )
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"Expected exactly two visible GPUs, found {torch.cuda.device_count()}. "
            "Run with CUDA_VISIBLE_DEVICES=2,3."
        )

    model_path = str(Path(args.model_path).resolve())
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    workflow = ModularPipeline.from_pretrained(model_path, local_files_only=True).blocks.get_workflow("t2va")

    text_manager = ComponentsManager()
    text_manager.enable_auto_cpu_offload(device="cuda:1")
    conditioner = workflow.sub_blocks.pop("text_encoder").init_pipeline(
        model_path,
        components_manager=text_manager,
    )
    conditioner.load_components(
        dtype=torch.bfloat16,
        pretrained_model_name_or_path=model_path,
        local_files_only=True,
    )

    generation_manager = ComponentsManager()
    generation_manager.enable_auto_cpu_offload(device="cuda:0")
    generator_pipeline = workflow.init_pipeline(
        model_path,
        components_manager=generation_manager,
    )
    generator_pipeline.load_components(
        dtype=torch.bfloat16,
        pretrained_model_name_or_path=model_path,
        local_files_only=True,
    )

    state = conditioner(prompt=args.prompt)
    results = generator_pipeline(
        state=state,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        generator=torch.Generator().manual_seed(args.seed),
        output=["videos", "audio", "sampling_rate"],
    )

    encode_video(
        results["videos"][0],
        fps=24,
        output_path=str(output_path),
        audio=results["audio"][0],
        audio_sample_rate=results["sampling_rate"],
    )
    print(output_path)


if __name__ == "__main__":
    main()
