#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export HF_HOME="${HF_HOME:-/data/chenjiayu/wenbiao_zhao/hf_home}"

uv run --frozen python scripts/h200_t2va.py "$@"
