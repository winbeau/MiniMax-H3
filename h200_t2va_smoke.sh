#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export HF_HOME="${HF_HOME:-/data/chenjiayu/wenbiao_zhao/hf_home}"

uv_bin="${UV_BIN:-/data/chenjiayu/.local/bin/uv}"

"$uv_bin" run --frozen python scripts/h200_t2va.py "$@"
