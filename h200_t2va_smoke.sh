#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export HF_HOME="${HF_HOME:-/data/chenjiayu/wenbiao_zhao/hf_home}"

uv_bin="${UV_BIN:-/data/chenjiayu/.local/bin/uv}"
text_encoder_device="${TEXT_ENCODER_DEVICE:-cpu}"
height="${HEIGHT:-256}"
width="${WIDTH:-448}"
num_frames="${NUM_FRAMES:-124}"
num_inference_steps="${NUM_INFERENCE_STEPS:-2}"

default_args=(
  --text-encoder-device "$text_encoder_device"
  --height "$height"
  --width "$width"
  --num-frames "$num_frames"
  --num-inference-steps "$num_inference_steps"
)
if [[ "${GROUP_OFFLOAD:-1}" == "1" ]]; then
  default_args+=(--group-offload)
fi

"$uv_bin" run --frozen python scripts/h200_t2va.py "${default_args[@]}" "$@"
