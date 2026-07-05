#!/bin/bash

uv run python -m attn_explorer.serve \
      --host 0.0.0.0 --port 5001 \
      --store data/vnbench/output_v4 \
      --videos data/vnbench/videos/v4