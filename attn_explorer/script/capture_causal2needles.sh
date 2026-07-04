#!/bin/bash
uv run python -m attn_explorer.capture \
      --video-dir videos/causal-2-needles \
      --prompt-from-json \
      --outdir debug_out_causal2needles
