# Debug attenzione entity→visivo in locale

Pipeline di `debug_attention.py` (forward → attention map entity→visivo →
overlay sui frame) eseguita **sulla macchina locale**, senza HPC e senza la
sua cache condivisa. I pesi di Qwen2.5-VL-3B vengono scaricati una volta sola
nella cache HuggingFace **locale** (non `/work/tesi_avalenza/hf`).

## Comando

```bash
HF_HOME=~/.cache/huggingface \
uv run python debug_attention.py \
    --video videos/paperella.mp4 \
    --entity duck \
    --double-frames \
    --dtype float32 \
    --device-map cpu \
    --nframes 16
```

In alternativa, dopo un `uv sync`, l'entry point è più corto:

```bash
HF_HOME=~/.cache/huggingface \
uv run debug_attention \
    --video videos/paperella.mp4 --entity duck --double-frames \
    --dtype float32 --device-map cpu --nframes 16
```

Equivalente con il flag dedicato invece della variabile d'ambiente:
`--hf-home ~/.cache/huggingface`.

## Perché questi argomenti

- `HF_HOME=~/.cache/huggingface` — cache locale (default HuggingFace). Sostituisci
  col path locale che preferisci; NON usare `/work/tesi_avalenza/hf` (è l'HPC).
- `--video videos/paperella.mp4` — il prompt è letto in automatico da
  `videos/paperella.txt` (stesso stem); `--entity duck` compare in quel testo.
- `--double-frames` — ogni cella temporale dell'attenzione ↔ un singolo frame.
- `--dtype float32` `--device-map cpu` — in locale di solito **non c'è GPU**:
  float32 su CPU evita gli op bf16 lenti/non supportati e forza il placement su CPU.
- `--nframes 16` — meno frame dei 64 dell'eval: su CPU il forward è lento, 16
  bastano per validare il code path e ispezionare la heatmap.

> Su CPU il forward resta lento (minuti) e la prima esecuzione scarica ~7 GB di
> pesi. Per i run "veri" usare l'HPC (con GPU e `HF_HOME=/work/tesi_avalenza/hf`).

## Output

In `debug_out/` (override con `--outdir`):

- `debug_duck.pt` — dump della heatmap `[t, grid_h, grid_w]` + metadati;
- `attended_frames/` — top-K frame con attenzione più alta + overlay.

Il dump si può ri-analizzare senza rifare il forward:

```bash
uv run python utils/attn_frames.py --tensor debug_out/debug_duck.pt --overlay
```
