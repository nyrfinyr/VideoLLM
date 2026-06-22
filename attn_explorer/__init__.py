"""Esplorazione dell'attenzione testo→visivo di Qwen2.5-VL, indipendente da `main.py`.

Parte separata dall'eval su dataset (`main.py`): cattura le mappe di attenzione
COMPLETE (entity-agnostiche) di uno o più video e le esplora a runtime.

- `attn_explorer.capture` — tool CLI di cattura → store su disco
  (`python -m attn_explorer.capture --video-dir <dir>`).
- `attn_explorer.serve`   — server Flask interattivo sullo store
  (`python -m attn_explorer.serve --store <dir>`).

I DTO frozen e le funzioni di overlay/ranking condivise stanno in
`utils.attn_core` (riusati anche dai metodi di `models.Qwen25VLAttention`).
"""
