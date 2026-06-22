#!/bin/bash
uv run python -m attn_explorer.capture \
      --video-dir videos/qv-highlights \
      --prompt-from-json \
      --outdir debug_out_2 \
      --double-frames