# Probe di throughput → stima del walltime degli sbatch

> **Stato** (2026-05-27): tutte e 4 le probe **misurate** — vedi
> [Risultati](#risultati). egoschema/mvbench/video_mme su L40S; lvbench (768f)
> su nodo `huber`. `model_latency` letti via API Weave (durate op `predict`).

Obiettivo: misurare il **tempo per-sample** (`model_latency`, secondi/sample)
di Qwen2.5-VL-3B su ogni benchmark, a **config di produzione**, per
dimensionare i `--time` degli sbatch di prod e decidere come gestire i job
pesanti (video_mme, lvbench).

Ancora storica: egoschema (config *Subset*, 500 sample) ≈ **2 h** su una GPU
16 GB col vecchio `fps=0.25` (~45 frame). Le probe rifanno la misura sulla
config attuale (`nframes` fissi) e sulle card consentite (24G/45G).

---

## Come sono fatte le probe

Ogni `scripts/sbatch/<dataset>-test.sbatch`:

- gira a **config PROD** (stessi `nframes` / `max_pixels` del run vero);
- processa un **subset rappresentativo** — `shuffle=true` (RNG seedato con
  `seed`, mischia *prima* di troncare) + `limit=N`. Lo shuffle è essenziale:
  i loader emettono in ordine (mvbench per task, lvbench per video), quindi i
  primi N senza shuffle sarebbero biased;
- **completa** l'eval (non viene killato a tempo) → Weave logga il summary
  con `model_latency.mean`;
- ha **Weave attivo**, `wandb.group=<model>-<dataset>-test`.

Knob rilevanti (in `conf/config.yaml`, applicati in `main.py`):

- `limit: null` — numero massimo di sample (tronca dopo loader + filtri);
- `shuffle: false` — mischia i sample prima di `limit` (irrilevante a
  `limit=null`: l'accuracy non cambia, ogni sample è indipendente).

| probe | frame (prod) | `limit` | `--time` | mem |
|:--|--:|--:|--:|--:|
| egoschema | 64 | 40 | 25 min | 16G |
| mvbench | 32 | 60 | 20 min | 16G |
| video_mme | 128 | 40 | 45 min | 16G |
| lvbench | 768 | 12 | 1h45 | 32G |

---

## Lanciare e leggere

```bash
mkdir -p logs
sbatch scripts/sbatch/egoschema-test.sbatch    # idem mvbench / video_mme / lvbench
```

Misura precisa = **`model_latency.mean`** (secondi/sample):

- **UI Weave**: progetto `<dataset>` → group `<model>-<dataset>-test`. Lì
  trovi anche **p50/p95** (usa la mediana: il 1° sample include il warmup
  CUDA e gonfia la media);
- oppure dal log: `grep -A6 'Evaluation summary' logs/<dataset>-test-<jobid>.out`.

Caveat:

- **1° sample = warmup** (init CUDA/cuDNN) → outlier; preferire la mediana.
- **Varianza intra-dataset**: video_mme (short/medium/long) e lvbench
  (durata video 1–2 h) hanno per-sample molto variabile → lo shuffle media le
  fasce, ma con `limit` piccolo la stima resta rumorosa.
- **Card**: il `constraint` mischia 24G (RTX6000/A5000) e 45G (A40/L40S).
  RTX6000 è la più lenta → worst-case; L40S la più veloce. Annotare quale
  card ha pescato (`nvidia-smi` a inizio log) accanto al `model_latency`.
- La probe **lvbench valida anche la VRAM** a 768 frame: se muore per **OOM**
  (non a fine eval) → abbassare `max_pixels`.

---

## Stima del tempo di prod

```
walltime_prod ≈ N_totali × model_latency   (+ ~1-2 min di load, trascurabile)
```

`N_totali` per dataset:

| dataset | N sample totali |
|:--|--:|
| egoschema (Subset) | 500 |
| mvbench | ~3800 (19 task × ~200) |
| video_mme | 2700 |
| lvbench | 1549 |

---

<a id="risultati"></a>
## Risultati

> Misura **2026-05-26**, tutte su **L40S 45G** (card più veloce del
> constraint). `model_latency` = durata media op `predict` (via API Weave);
> `prod stimato` = N_totali × mean. ⚠️ Su **RTX6000-24G** (la più lenta
> consentita) aspettarsi **~2-3×** → vedi `--time` consigliato (con margine).

| dataset | card | model_latency (mean / med / p95) | prod stimato | `--time` consigliato | note |
|:--|:--|--:|--:|--:|:--|
| egoschema | L40S | 4.0 / 3.8 / 6.4 s | ~0.6 h (500×4.0) | 02:00:00 | bassa varianza |
| mvbench | L40S | 2.2 / 1.1 / 9.8 s | ~2.1 h (3800×2.2) | 06:00:00 | alta varianza fra task |
| video_mme | L40S | 12.4 / 13.1 / 20.1 s | ~9.3 h (2700×12.4) | 14:00:00 ¹ | varianza durate short/med/long |
| lvbench | huber ² | 75.7 / 82.6 / 109 s | ~32.6 h (1549×75.7) | **job array** ³ | 768f, no OOM; acc 7/12=58% |

¹ video_mme su L40S sta in ~9-10 h, ma su RTX6000-24G salirebbe a ~25 h
(oltre il singolo job): per il prod **pinnare A40/L40S** (constraint
`gpu_A40_45G|gpu_L40S_45G`) e tenere 14h, oppure job array.

² Card non registrata in wandb (solo nello SLURM `.out`: header `nvidia-smi`);
nelle note del run solo il nodo `huber`. Verificare card per capire se i
75.7 s sono best- o worst-case; a constraint misto assumere il peggio.

³ **Singolo job impossibile**: 32.6 h è già il best-case (mean, presumibilmente
su card veloce) e sfora il `--time 23:00:00`; su RTX6000-24G salirebbe a
~65-95 h. Serve **job array** (sharding) — vedi [Decisioni](#decisioni-aperte-post-misura).
Stima per shard: pin A40/L40S + `--array=0-3` → ~388 sample/shard × 75.7 s
≈ **8.2 h**, `--time 12:00:00` con margine.

---

## Walltime prod attuali (provvisori, da rivedere coi dati)

In `scripts/sbatch/<dataset>.sbatch`:

| dataset | `--time` attuale | verdetto coi dati reali (L40S) |
|:--|--:|:--|
| egoschema | 04:00:00 | ✅ abbondante (~0.6 h L40S, ~1.5 h worst-case) → si può scendere a 02:00:00 |
| mvbench | 14:00:00 | ⬇️ sovradimensionato (~2.1 h) → 06:00:00 basta |
| video_mme | 08:00:00 | ⚠️ **insufficiente**: ~9.3 h già su L40S → 14:00:00 + pin A40/L40S |
| lvbench | 23:00:00 | ❌ **insufficiente**: ~32.6 h già best-case (768f) → singolo job impossibile, serve **job array** |

## Decisioni aperte (post-misura)

Numeri reali alla mano:

- **egoschema**: ok singolo job (~0.6 h L40S). `--time` 02:00:00.
- **mvbench**: ok singolo job (~2.1 h). `--time` 06:00:00.
- **video_mme** (~9.3 h L40S, ~25 h RTX6000): **borderline**. Pin A40/L40S +
  `--time 14:00:00` regge; altrimenti job array.
- **lvbench** (~32.6 h best-case): **job array obbligatorio** — nessuna card
  consentita lo fa in un singolo job entro 23 h.

Opzioni residue (in ordine di preferenza per restare fedeli al paper):

1. **Job array (sharding)** — knob `offset`/`shard` in `main.py` + `#SBATCH
   --array`, dataset spezzato in N job. Mantiene il setup paper. Necessario per
   lvbench; raccomandato per video_mme. *(Da implementare: lo shard knob in
   `main.py` non c'è ancora — solo `limit`/`shuffle`.)*
2. **Riduci `nframes` prod** — singolo job fattibile, ma ci si allontana dal
   paper (lvbench a 768f è esplicitamente il setup paper).
3. **Solo card veloci A40/L40S** + walltime alto — basta per video_mme,
   **non** per lvbench (32.6 h > 23 h anche su card veloce).
