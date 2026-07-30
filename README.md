# Video LLM

Checkpoint dello stato esperimenti — aggiornato a ogni push. 

## Struttura del repo

Layout piatto (niente `src/` — nessun pacchetto distribuito, tutto gira
via `uv run python main.py` dal root, vedi `CLAUDE.md` per i dettagli).

| Cartella | Contenuto |
|---|---|
| `main.py` | entrypoint eval (compone config → dataset → strategy → modello) |
| `conf/` | `config.yaml` unico con tutti i preset (model/dataset/strategy/wandb) |
| `evals/` | un modulo per benchmark (EgoSchema, MVBench, Video-MME, LVBench, Causal2Needles) |
| `strategies/` | logica di campionamento/ricampionamento frame (`uniform`, `entropy_shortcut`, `entropy_attention_resample`) |
| `models/` | wiring dei VLM (Qwen2.5-VL, Qwen3-VL, variante con cattura attenzione) |
| `utils/` | helper condivisi (config loader, MCQ parsing, attention core) |
| `attn_explorer/` | tool standalone cattura + esplorazione interattiva delle mappe di attenzione |
| `fetch/` | script di prefetch dei dump dataset (girano su nodo con internet) |
| `scripts/` | sbatch SLURM + script di sweep/lancio |
| `docs/` | analisi, piani, diario — cronologia del ragionamento dietro le scelte |
| `data/`, `outputs/`, `logs/`, `wandb/`, `debug_out/` | scratch locale, gitignored |

## Wired

- **Modelli:** 
  * [Qwen2.5-VL-3B](https://github.com/QwenLM/Qwen2.5-VL)
  * [Qwen3-VL-2B](https://github.com/QwenLM/Qwen3-VL)
  * [Qwen3-VL-4B](https://github.com/QwenLM/Qwen3-VL)

  Ognuno ha anche una variante `*_attn` (`qwen2_5_vl_3b_attn`,
  `qwen3_vl_2b_attn`, `qwen3_vl_4b_attn`) che monta la cattura
  dell'attenzione e implementa `SupportsSignals`: è quella richiesta dalle
  strategy signal-driven. Differenze Qwen3 (token visivi non contigui, sink
  dims, timestamp nel prompt) in `docs/qwen3vl_signals.md`.
- **Dataset:** 
  * [EgoSchema](https://github.com/egoschema/EgoSchema)
  * [MVBench](https://github.com/OpenGVLab/Ask-Anything)
  * [Video-MME](https://github.com/BradyFU/Video-MME)
  * [LVBench](https://github.com/THUDM/LVBench)

## Baseline (accuracy MCQ)

Greedy (`generation=precise`), seed=42, fullset. 
Numeri del paper Qwen2.5-VL da
[arxiv:2502.13923](https://arxiv.org/abs/2502.13923), Tab. 8.

| Dataset             | `nframes` | Qwen2.5-VL-3B<br>(paper) | Qwen2.5-VL-3B<br>(ours)                                                              | Δ vs paper | Qwen3-VL-2B | Qwen3-VL-4B |
|---------------------|----------:|-------------------------:|--------------------------------------------------------------------------------------|-----------:|-------------|-------------|
| EgoSchema           |        64 |                     64.8 | [62.60](https://wandb.ai/alesvale97-unimore/egoschema/runs/y6ysm2tz) (313/500)       |       −2.2 | —           | —           |
| MVBench             |        32 |                     67.0 | [64.83](https://wandb.ai/alesvale97-unimore/mvbench/runs/srzjo9j9) (2430/3748\*)     |       −2.2 | —           | —           |
| Video-MME (no subs) |       128 |                     61.5 | [59.93](https://wandb.ai/alesvale97-unimore/video_mme/runs/p7srmrkl) (1618/2700)     |       −1.6 | —           | —           |
| LVBench             |       768 |                     43.3 | [43.00](https://wandb.ai/alesvale97-unimore/lvbench/groups/qwen2_5_vl_3b-lvbench) (666/1549) — vedi nota | −0.3 | —           | —           |

\* MVBench: 52/3800 sample senza `pred` parseabile (4 OOM + 48 parser MCQ).

**LVBench**: eval in 3 shard SLURM array, accuracy *pooled* sull'intero
dataset (666/1549 sample). Lo shard 2 originale era finito su una GPU
24 GiB (`nullazzo`) → 1032 OOM su 516 sample, accuracy 0%; è stato
rilanciato su GPU ≥ 40 GiB
([`sk30qdp6`](https://wandb.ai/alesvale97-unimore/lvbench/runs/sk30qdp6),
209/516 = 40.50%) e ricombinato con gli shard 0
([`45bdgea9`](https://wandb.ai/alesvale97-unimore/lvbench/runs/45bdgea9),
230/517) e 1
([`usuy6en7`](https://wandb.ai/alesvale97-unimore/lvbench/runs/usuy6en7),
227/516). In linea col paper (−0.3 punti).

**Nota su tutti i numeri Qwen2.5 di questa pagina** — il preprocessing non
passa al processor i `video_metadata` reali, quindi `second_per_grid_ts` vale
0.083 per qualunque video e `get_rope_index` lo tronca a 0: le celle video
finiscono tutte sulla stessa posizione temporale M-RoPE. Riguarda ogni run
qui riportata; è il regime in cui sono state prodotte, quindi resta il
default (knob `model.pass_video_metadata`, default `false`, da alzare solo
per una run di controllo). Misura e implicazioni in
`docs/qwen3vl_signals.md`.

## Notifiche Telegram dei job SLURM

Ogni job sottomesso tiene **un solo messaggio** su Telegram, modificato in
place: `🕓 PENDING` alla sottomissione → `🏃 RUNNING` quando SLURM lo schedula
(lo script sbatch parte esattamente in quel momento, nessun polling di
`squeue`) → `✅/❌/⏱` a fine job con **tutte** le metriche di
`wandb.run.summary`. Le metriche non sono elencate da nessuna parte: chi ne
aggiunge una in `utils/obs.py` se la ritrova su Telegram senza toccare altro.

Setup, una volta sola sul cluster:

1. crea il bot con [@BotFather](https://t.me/BotFather) → token;
2. scrivi un messaggio qualsiasi al bot, poi apri
   `https://api.telegram.org/bot<TOKEN>/getUpdates` e leggi `message.chat.id`;
3. `~/.config/telegram-notify.env` (`chmod 600`, **fuori dal repo**):
   ```sh
   TG_BOT_TOKEN=123456:AAH...
   TG_CHAT_ID=987654321
   ```
4. verifica che il firewall non filtri Telegram **dal nodo di calcolo** (che
   wandb funzioni prova solo che c'è egress HTTPS, non che ci sia questo host):
   ```sh
   srun --account=tesi_avalenza --partition=all_usr_prod --time=00:02:00 \
     bash -c 'source ~/.config/telegram-notify.env
              curl -sS --max-time 10 "https://api.telegram.org/bot$TG_BOT_TOKEN/getMe"'
   ```

Poi si sottomette con `scripts/tgsbatch` al posto di `sbatch` (stessi
argomenti, stesso output, stesso exit code); i launcher di sweep lo usano già.
Sottomettere con `sbatch` diretto resta legittimo: si perde solo lo stato
PENDING, il resto del ciclo arriva lo stesso. **Senza il file di configurazione
tutto è no-op** e i job girano esattamente come prima.

| Voglio | Come |
|---|---|
| sottomettere in silenzio | `SBATCH_BIN=sbatch ./scripts/sweep_video_mme_thresholds.sh` |
| disattivare tutto per un job | `TELEGRAM_NOTIFY_ENV=/dev/null sbatch ...` |
| notificare da uno script nuovo | `source scripts/lib/notify.sh && tg_job_start` dopo il `cd` nel repo |

## Sweep `entropy_attention_resample` (Video-MME, in corso)

Subset comune a tutte le righe: `limit=250`, `shuffle=true`, `seed=42`
(stesso subset per ogni combinazione — shuffle deterministico, vedi
`scripts/sweep_video_mme_thresholds.sh`). Grid su `entropy_threshold` ×
`concentration_threshold`, più un controllo `uniform` di baseline sullo
stesso subset. `Δ vs uniform` = accuracy − accuracy del controllo uniform
**su questo stesso subset** (non il 59.93% della tabella Baseline sopra,
che è sull'intero set da 2700 sample — varianza campionaria diversa, non
confrontabile direttamente). `resampled` = % sample che hanno superato il
gate d'entropia (`answer_entropy > entropy_threshold`); `fixed`/`broken` =
sample ricampionati che il resampling ha rispettivamente corretto
(❌→✅) o rovinato (✅→❌) rispetto al pass 1, dal join per-sample in
`docs/analisi_winloss_resampling_video_mme.md`. `—` = 0 predizioni valide
per OOM, vedi nota.

| strategy                   | entropy_threshold | concentration_threshold | accuracy         | Δ vs uniform | resampled | fixed | broken | net | run |
|-----------------------------|-------------------:|--------------------------:|------------------:|-------------:|-----------:|------:|-------:|----:|-----|
| uniform (controllo)         |                   — |                          — | [62.80](https://wandb.ai/alesvale97-unimore/video_mme/runs/ktvjcyye) (157/250) |            — | — | — | — | — | [`ktvjcyye`](https://wandb.ai/alesvale97-unimore/video_mme/runs/ktvjcyye) |
| entropy_attention_resample  |                 0.3 |                         20 | [65.60](https://wandb.ai/alesvale97-unimore/video_mme/runs/r6khe103) (164/250) |        +2.80 | 77.6% (194/250) | 13 | 6 | +7 | [`r6khe103`](https://wandb.ai/alesvale97-unimore/video_mme/runs/r6khe103) |
| entropy_attention_resample  |                 0.3 |                         30 | [65.60](https://wandb.ai/alesvale97-unimore/video_mme/runs/rsw3fq7d) (164/250) |        +2.80 | 77.6% (194/250) | 13 | 6 | +7 | [`rsw3fq7d`](https://wandb.ai/alesvale97-unimore/video_mme/runs/rsw3fq7d) |
| entropy_attention_resample  |                 0.3 |                         45 | [65.60](https://wandb.ai/alesvale97-unimore/video_mme/runs/em2fzfr1) (164/250) |        +2.80 | 77.6% (194/250) | 13 | 6 | +7 | [`em2fzfr1`](https://wandb.ai/alesvale97-unimore/video_mme/runs/em2fzfr1) |
| entropy_attention_resample  |                 0.5 |                         20 | [65.60](https://wandb.ai/alesvale97-unimore/video_mme/runs/86jyn8n7) (164/250) |        +2.80 | 71.2% (178/250) | 13 | 6 | +7 | [`86jyn8n7`](https://wandb.ai/alesvale97-unimore/video_mme/runs/86jyn8n7) |
| entropy_attention_resample  |                 0.5 |                         30 | [65.60](https://wandb.ai/alesvale97-unimore/video_mme/runs/yp7r9ziy) (164/250) |        +2.80 | 71.2% (178/250) | 13 | 6 | +7 | [`yp7r9ziy`](https://wandb.ai/alesvale97-unimore/video_mme/runs/yp7r9ziy) |
| entropy_attention_resample  |                 0.5 |                         45 | [65.60](https://wandb.ai/alesvale97-unimore/video_mme/runs/mb7cgbq5) (164/250) |        +2.80 | 71.2% (178/250) | 13 | 6 | +7 | [`mb7cgbq5`](https://wandb.ai/alesvale97-unimore/video_mme/runs/mb7cgbq5) |
| entropy_attention_resample  |                 0.7 |                         20 | — (0/250, OOM — vedi nota) |            — | — | — | — | — | [`lgs1jrxb`](https://wandb.ai/alesvale97-unimore/video_mme/runs/lgs1jrxb) |
| entropy_attention_resample  |                 0.7 |                         30 | [66.00](https://wandb.ai/alesvale97-unimore/video_mme/runs/y6u5k1pq) (165/250) |        +3.20 | 65.2% (163/250) | 13 | 5 | +8 | [`y6u5k1pq`](https://wandb.ai/alesvale97-unimore/video_mme/runs/y6u5k1pq) |
| entropy_attention_resample  |                 0.7 |                         45 | — (0/250, OOM — vedi nota) |            — | — | — | — | — | [`tzhm99kn`](https://wandb.ai/alesvale97-unimore/video_mme/runs/tzhm99kn) |
| entropy_attention_resample  |                 1.0 |                         20 | [66.00](https://wandb.ai/alesvale97-unimore/video_mme/runs/hmpib0w2) (165/250) |        +3.20 | 59.2% (148/250) | 13 | 5 | +8 | [`hmpib0w2`](https://wandb.ai/alesvale97-unimore/video_mme/runs/hmpib0w2) |
| entropy_attention_resample  |                 1.0 |                         30 | [66.00](https://wandb.ai/alesvale97-unimore/video_mme/runs/igbsy8cp) (165/250) |        +3.20 | 59.2% (148/250) | 13 | 5 | +8 | [`igbsy8cp`](https://wandb.ai/alesvale97-unimore/video_mme/runs/igbsy8cp) |
| entropy_attention_resample  |                 1.0 |                         45 | — (0/250, OOM — vedi nota) |            — | — | — | — | — | [`enb7u9ui`](https://wandb.ai/alesvale97-unimore/video_mme/runs/enb7u9ui) |

Le tre combinazioni con lo stesso `entropy_threshold` sono byte-per-byte
identiche (stesso set di sample fixed/broken, non solo lo stesso
conteggio) — `concentration_threshold` non misura alcun effetto in
{20,30,45} su questo dataset (dettagli in
`docs/analisi_winloss_resampling_video_mme.md`).

La riga et0.7/ct30 è la run di verifica pre-sweep (post-fix OOM
`empty_cache`, vedi `models/qwen_attn.py`) — non fa parte dello sweep group
(`qwen2_5_vl_3b_attn-sweep-video_mme-test`) ma condivide lo stesso subset e
config, quindi il numero è valido e riportato qui.

**OOM totale su 3 combinazioni, legato all'host non alle soglie** —
`et0.7-ct20`, `et0.7-ct45` ed `et1.0-ct45` hanno **0/250 predizioni
valide**: ogni singola `predict_and_score` fallisce con
`torch.OutOfMemoryError` dentro `full_visual_attention`
(`models/qwen_attn.py:221`, il forward di prefill per il segnale, chiamato
da `generate_with_signals` — **prima** di qualunque logica di
resampling/soglia). `Evaluation.evaluate` di Weave cattura l'eccezione per
sample invece di far crashare il job, quindi wandb marca il run
`finished` senza errore visibile — solo `mcq_accuracy` risulta assente
perché calcolato su zero sample validi (attenzione: un run `finished` con
`model_latency_mean`/`n_samples` ma senza `mcq_accuracy` nel summary va
sempre controllato via Weave, `client.get_call(<id>).output`, prima di
fidarsi — non basta guardare lo stato wandb).

**Non è un effetto di `concentration_threshold=45`**: `et0.3-ct45` ed
`et0.5-ct45` sono riusciti senza problemi. È l'**host** — le 3 run fallite
sono girate **tutte e sole** su `nullazzo` (Quadro RTX 6000 24G); `huber`
(RTX A5000, anch'essa 24G) è invece andata bene due volte
(`et1.0-ct20`/`et1.0-ct30`) — dettagli e tabella host/GPU completa in
`docs/analisi_winloss_resampling_video_mme.md`. Il fix `empty_cache`
(commit `d826301`) non basta a evitarlo su quel nodo con questo carico
(128 frame + capture attention completa su tutti i layer/head). Il full
eval (`scripts/sbatch/video_mme-signal.sbatch`) è già pinnato a sole GPU
45G (A40/L40S), quindi esclude comunque `nullazzo`.

## Ablation entropy/shift/zoom (Video-MME)

7 arm su Video-MME intero (2700 sample, `model=qwen2_5_vl_3b_attn`,
`dataset.nframes=24`, job array a 3 shard SLURM,
`scripts/sbatch/ablation-videomme/`, gruppo wandb
[`video_mme-ablation`](https://wandb.ai/alesvale97-unimore/video_mme/groups/video_mme-ablation),
vedi `docs/future-work/output_riunione_2207.md`). `baseline_24` è il
riferimento `uniform` (regime normale, merge a coppie); `baseline_24double`
è lo stesso `uniform` ma con timestamp raddoppiati (24 celle da 1 frame
invece di 12 da 2). Gli altri quattro arm sono signal-driven e condividono
lo stesso gate d'entropia: tre ricampionano i frame
(`entropy_attention_resample`, strategie `phase_shift`/`zoom_peak`/router),
il quarto no — `highlight_top1_24` lascia i frame invariati e agisce solo
sul prompt (`attention_highlight`, vedi sotto). `random_top1_24` è il suo
controllo: identico in tutto tranne che indica una cella a caso.
Tutti e 21 i run dei 7 arm (7 × 3 shard) sono `finished`. Accuracy *pooled* sull'intero arm
(Σ corretti / Σ campioni, non media degli shard — stesso metodo di
LVBench sopra).

Colonne: accuracy pooled complessiva, `Δ` contro il riferimento
`baseline_24` (tutti gli arm si confrontano con quello, non con
`baseline_24double`), per tipologia video (short/medium/long,
breakdown emesso da `evals/base.py` via `BREAKDOWN_KEYS`), `% resampling`
(quota di sample su cui è scattato il secondo passaggio, cioè
`answer_entropy > entropy_threshold` al pass 1 — per `highlight_top1_24`
il secondo passaggio c'è comunque ma non ricampiona nulla, cambia solo il
prompt), e accuracy condizionata al gate d'entropia
(`acc | high_ans_entropy` = sample su cui il gate è scattato,
`acc | low_ans_entropy` = sample accettati al pass 1 — vedi nota in
`evals/base.py:96-104`), e — solo per
`shift_zoom_router_24`, l'unico arm dove il ramo è deciso sample-per-
sample invece che fissato per costruzione — lo split fra i due rami di
resampling (`ramo phase_shift`/`ramo zoom_peak`: quota sul totale
ricampionato e accuracy condizionata). `baseline_24`/`baseline_24double`
sono `uniform`: non hanno gate, colonne assenti (`—`).

| arm                  | pooled | Δ vs baseline_24 | short | medium | long | % resampling | acc \| high_ans_entropy | acc \| low_ans_entropy | ramo phase_shift | ramo zoom_peak |
|-----------------------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_24           | **55.04%** (1486/2700) | — (riferimento) | 64.33% (579/900) | 52.00% (468/900) | 48.78% (439/900) | — | — | — | — | — |
| baseline_24double     | **56.93%** (1537/2700) | +1.89 | 68.22% (614/900) | 55.22% (497/900) | 47.33% (426/900) | — | — | — | — | — |
| entropy_shift_24      | 55.63% (1502/2700) | +0.59 | 66.11% (595/900) | 53.00% (477/900) | 47.78% (430/900) | 73.15% (1975/2700) | 43.59% (861/1975) | 88.41% (641/725) | 100% (forzato) | — |
| entropy_zoom_24       | 54.59% (1474/2700) | −0.44 | 63.33% (570/900) | 52.44% (472/900) | 48.00% (432/900) | 72.93% (1969/2700) | 42.10% (829/1969) | 88.24% (645/731) | — | 100% (forzato) |
| shift_zoom_router_24  | 55.81% (1507/2700) | +0.78 | 66.00% (594/900) | 53.33% (480/900) | 48.11% (433/900) | 72.93% (1969/2700) | 43.93% (865/1969) | 87.82% (642/731) | 96.95% (1909/1969), acc 43.74% | 3.05% (60/1969), acc 50.00%\* |
| highlight_top1_24     | 54.63% (1475/2700) | −0.41 | 63.22% (569/900) | 52.22% (470/900) | 48.44% (436/900) | 73.07% (1973/2700) | 42.17% (832/1973) | 88.45% (643/727) | — | — |
| random_top1_24        | 54.78% (1479/2700) | −0.26 | 62.89% (566/900) | 53.11% (478/900) | 48.33% (435/900) | 73.00% (1971/2700) | 42.42% (836/1971) | 88.20% (643/729) | — | — |

\* N piccola (60 sample) — indicativo, non conclusivo. Il salto
~42-44% → ~88% sul gate non è il gate che "funziona meglio quando non
scatta": è un effetto di selezione — il gate manda a ricampionamento i
sample con attenzione dispersa (entropia alta), che nel dataset sono
anche quelli con domande intrinsecamente più difficili (video
lunghi/complessi); i sample accettati al pass 1 sono i più facili per
costruzione. Non è quindi un confronto valido tra "il gate aiuta" vs "il
gate non aiuta" — servirebbe un controllo uniform ristretto allo stesso
sottoinsieme di sample (`high_ans_entropy`) per isolare l'effetto del
ricampionamento dalla difficoltà intrinseca del sample.

Sul full dataset: `entropy_shift_24` (+0.59) e `shift_zoom_router_24`
(+0.78) sono debolmente sopra `baseline_24`; `entropy_zoom_24` è
**sotto** (−0.44) — lo zoom a singolo picco forzato, da solo, non recupera
sample rispetto a non ricampionare affatto. Nessuno dei tre batte
`baseline_24double` (56.93%), il controllo compute-matched.

Il gate d'`highlight_top1_24`/`random_top1_24` scatta sul 73.07%/73.00%
dei sample (1973 e 1971 su 2700) — in linea col ~73% degli altri arm
signal-driven, come atteso (stesso pass 1, stesso gate).

**Il risultato che conta è il confronto fra i due**: `highlight_top1_24`
(54.63%, −0.41 vs `baseline_24`) e `random_top1_24` (54.78%, −0.26) sono
indistinguibili entro il rumore campionario, e il controllo casuale è
addirittura leggermente sopra la versione guidata dall'attenzione
(+0.15). Nessuno dei due batte `baseline_24`. Indicare al modello dove
guardare non aiuta più quando l'indicazione viene dal picco di attenzione
rispetto a quando è estratta a caso.

Quel nulla ammetteva però due cause opposte — **(A)** il puntatore
funziona ma punta male, **(B)** il modello non lo esegue affatto — con
conseguenze pratiche inverse: in (A) migliorare il segnale ha senso, in
(B) è tempo buttato, perché ogni miglioria agisce su *dove* si punta. La
sonda della sezione seguente separa le due, ed è risultata **(A)**.

Non è stata ancora fatta un'analisi win/loss per-sample (fixed/broken vs
`baseline_24`, come in `docs/analisi_winloss_resampling_video_mme.md` per
lo sweep `entropy_attention_resample`) per escludere che i due arm
recuperino/rompano sample diversi nonostante l'accuracy pooled simile.

### Sonda di sensibilità del canale-prompt (`antipeak`)

> Non è un arm di accuracy e **non va letta nella tabella sopra**: gira su
> un subset da 500 sample (`shuffle=true`, `seed=42`, `limit=500`) invece
> che sui 2700 interi, quindi non è confrontabile con `baseline_24` né con
> gli altri arm. Razionale completo in
> `docs/sonda_sensibilita_puntatore_antipeak.md`, sbatch in
> `scripts/sbatch/ablation-videomme/antipeak_probe_24.sbatch`.

**A cosa serve.** Risponde alla domanda a monte di tutte le altre: *il
modello è capace di ricevere un'indicazione temporale via prompt?* Il modo
di scoprirlo non è puntare bene (servirebbe un ground truth temporale che
Video-MME non ha), ma puntare **di proposito nel posto sbagliato** — la
cella più lontana nel tempo dal picco (`cell_select=antipeak`) — e vedere
se si rompono i sample che il modello stava già risolvendo. È il motivo
per cui si legge solo sui corretti al pass 1 (`pred_pass1`, loggato dalla
strategy e incrociato con `answer` nello scorer): lì si parte da ~99% e lo
spazio per peggiorare c'è tutto, mentre `random_top1_24` girava col gate
aperto, cioè sul ~73% di sample dove l'accuracy è ~42% e non c'era niente
da rompere.

L'interpretazione non dipende da quanto è buono il picco: se è informativo
l'antipicco è sbagliato, se è solo posizionale l'antipicco è comunque
arbitrario — in entrambi i casi **nessun danno ⟹ il puntatore non viene
eseguito**.

Due arm appaiati sullo stesso subset, gruppo wandb
[`video_mme-antipeak-probe`](https://wandb.ai/alesvale97-unimore/video_mme/groups/video_mme-antipeak-probe).
Il controllo non è un di più: `pred_pass1` viene dalla softmax ristretta
alle lettere e `pred` dal parse di `generate()`, due decoding diversi,
quindi una quota di "corretto al pass 1, sbagliato al pass 2" esiste anche
**senza** puntatore, e va sottratta.

| arm | intervento | corretti al pass 1 | rotti al pass 2 | acc \| corretti al pass 1 | accuracy | run |
|---|---|---:|---:|---:|---:|---|
| `nopointer` (controllo) | nessuno (gate mai aperto) | 296/500 | **1** (0.34%) | **99.66%** (295/296) | 59.20% (296/500) | [`n0f2lrif`](https://wandb.ai/alesvale97-unimore/video_mme/runs/n0f2lrif) |
| `antipeak` | puntatore sulla cella più lontana dal picco | 294/500 | **20** (6.80%) | **93.20%** (274/294) | 57.00% (285/500) | [`c69hvw7w`](https://wandb.ai/alesvale97-unimore/video_mme/runs/c69hvw7w) |

```
danno = 99.66% − 93.20% = 6.46 pp        z = 4.24, p < 0.0001
```

**Il canale-prompt è vivo.** Indicare la parte sbagliata del video rompe
~1 sample su 15 fra quelli già corretti, contro un pavimento di controllo
allo 0.34%: l'effetto è quasi interamente attribuibile al puntatore, non
alla discrepanza fra i due decoding. Conferma dal lato opposto: sui
sample già sbagliati al pass 1 l'antipicco ne cambia 11/206 (5.34%) contro
1/204 (0.49%) del controllo. Rompe 20 e ripara 11 — netto −9 sample,
−2.20 pp di accuracy complessiva: un'istruzione sbagliata danneggia più di
quanto aiuti, che è la firma dell'obbedienza.

**Cosa implica per gli arm sopra.** Se il modello esegue il puntatore e
quello guidato dall'attenzione produce zero effetto netto, allora il picco
d'attenzione vale quanto una cella a caso — coerente col probe da 40
sample ([`2dodrndg`](https://wandb.ai/alesvale97-unimore/video_mme/runs/2dodrndg)):
picco sull'ultima cella nel 28% dei casi (3.4× il livello di caso), celle
estreme nel 40% (atteso 17%), `top1_pct` mediano 15.7% contro un livello
di caso dell'8.3%. Il meccanismo plausibile è la recency causale — le
righe-query sono i token della domanda, che stanno dopo il blocco visivo,
quindi l'ultima cella è la più vicina a prescindere dal contenuto; il
filtro sink non lo corregge, perché agisce su un effetto diverso.

Il filone quindi **non è chiuso, ma va spostato dal canale al segnale**:
la leva sono la risoluzione del puntatore (12 celle sono ~50 s su un video
long — da qui l'arm double-frame) e la qualità del picco (righe-query
ristrette ai token dell'entità, `question_rows` in `utils/attn_core.py`).

*Caveat.* 6.46 pp è il costo di puntare **malissimo**, e non implica
simmetricamente che puntare bene guadagni altrettanto: dice che il canale
ha banda, non quanta se ne può estrarre. Inoltre i due arm sono finiti su
GPU diverse (L40S vs Quadro RTX 6000), quindi il pass 1 non è identico
bit-per-bit — si vede nei conteggi di corretti al pass 1 (296 vs 294,
2 sample di oscillazione). Irrilevante a questa scala d'effetto, ma è il
motivo per cui le due righe non partono dallo stesso denominatore.

**Cosa rappresenta ogni `arm` (colonna della tabella):**

- **`baseline_24`** — riferimento: 24 frame campionati uniformemente
  lungo tutto il video, nessuna logica adattiva, un solo passaggio del
  modello per sample. È il punto zero rispetto a cui si misura ogni
  altro arm (colonna `Δ vs baseline_24`).
- **`baseline_24double`** — stesso campionamento uniforme di
  `baseline_24`, ma ogni frame viene mostrato al modello due volte
  invece di una. Non è una strategia adattiva: serve a isolare l'effetto
  di "più token visivi a parità di frame osservati" da quello degli arm
  successivi, che di fatto raddoppiano i token visivi solo sui sample
  ricampionati — un confronto a compute comparabile.
- **`entropy_shift_24`** — a partire da `baseline_24`, il modello fa un
  primo passaggio (pass 1) e si misura quanto è incerta la sua risposta
  (entropia della risposta). Se l'incertezza supera una soglia, si rifà
  la domanda un secondo passaggio (pass 2) dopo aver **spostato** la
  finestra temporale di frame osservata (stessa ampiezza, punto di
  osservazione diverso). In questo arm il secondo passaggio, quando
  scatta, è sempre di questo tipo (shift).
- **`entropy_zoom_24`** — stesso meccanismo di gate (pass 1 + soglia
  sull'incertezza), ma quando scatta il secondo passaggio **restringe**
  la finestra temporale ("zoom") attorno al singolo istante del video su
  cui il modello aveva concentrato l'attenzione al pass 1, invece di
  spostarla.
- **`shift_zoom_router_24`** — stesso gate d'incertezza, ma la scelta fra
  "shift" e "zoom" non è fissata a priori: viene decisa automaticamente,
  sample per sample, in base a quanto è concentrata l'attenzione del
  modello al pass 1 — molto concentrata su un istante → zoom, dispersa su
  più istanti → shift.
- **`highlight_top1_24`** — stesso gate d'incertezza dei tre precedenti,
  ma il secondo passaggio **non cambia i frame**: il modello rivede
  esattamente gli stessi 24, e l'unica differenza è una frase aggiunta in
  testa al prompt che gli dice dove guardare, del tipo *"Focus on the part
  of the video between 880 and 1100 seconds: that is where the visual
  evidence relevant to the question is"*. L'intervallo è quello della
  cella su cui l'attenzione del pass 1 si era concentrata di più (una
  sola, `top_k=1`), tolti prima i token-sink.

  Serve a separare due cose che negli arm precedenti sono confuse: quando
  shift o zoom migliorano una risposta, è perché il modello ha visto
  **frame nuovi** o perché è stato **ridiretto** su quelli che già aveva?
  Qui informazione visiva nuova non ce n'è per costruzione, quindi
  qualunque guadagno viene solo dal secondo. Il rovescio è che se il picco
  d'attenzione del pass 1 era sbagliato, shift e zoom possono comunque
  correggersi guardando altrove, mentre qui il puntatore rinforza
  l'errore: il confronto interessante è quanti sample recupera contro
  quanti ne rompe rispetto a `entropy_zoom_24`, che sui sample gated
  seleziona la stessa identica cella — uno la isola, l'altro la segnala.

  È l'adattamento al video di Look Twice (arXiv:2604.01280, codice in
  `lot/`), che sul lato immagine localizza l'evidenza con l'attenzione e
  poi la evidenzia nel prompt invece di ricampionare. Dettagli e
  divergenze dal paper in `strategies/attention_highlight.py`.
- **`random_top1_24`** — controllo di `highlight_top1_24`: identico in
  tutto (stesso gate, stessa frase, stesso costo) tranne che la parte di
  video indicata è **estratta a sorte** invece di venire dal picco di
  attenzione. Serve ad attribuire l'eventuale guadagno: se indicare un
  pezzo a caso rende quanto indicare quello scelto dall'attenzione, allora
  l'attenzione non sta contribuendo e si sta misurando solo l'effetto di
  dare al modello un'istruzione di focalizzazione, qualunque essa sia.

  Non è uno scrupolo formale. Un probe da 40 sample
  ([`2dodrndg`](https://wandb.ai/alesvale97-unimore/video_mme/runs/2dodrndg))
  mostra che il picco di attenzione è in buona parte **posizionale**: cade
  sull'ultima delle 12 celle nel 28% dei casi (3.4× il livello di caso) e
  su una delle due celle estreme nel 40% (atteso 17%), con `top1_pct`
  mediano al 15.7% contro un livello di caso dell'8.3%. Il meccanismo
  plausibile è la recency causale — le righe-query sono i token della
  domanda, che stanno dopo il blocco visivo, quindi l'ultima cella è la
  più vicina e riceve più attenzione a prescindere dal contenuto; il
  filtro sink non lo corregge, perché agisce su un effetto diverso. È
  anche una spiegazione candidata per `entropy_zoom_24`, che usa la stessa
  `peak_cell` per zoomare ed è l'unico arm sotto la baseline.

  ⚠️ Questo arm risponde alla domanda "l'attenzione contribuisce?", non a
  "il modello ascolta il puntatore?". Il fatto che `random ≈ attention`
  sembrava suggerire di no, ma è una lettura che il disegno non autorizza:
  girando col gate aperto, il puntatore cadeva su sample al ~42% di
  accuracy, dove un'istruzione dannosa non ha spazio per fare danno
  misurabile. La sonda `antipeak` (sezione sopra) mostra che il modello
  **esegue** il puntatore: quindi il nulla di questo confronto va letto
  come "il segnale è cattivo", non come "il canale è morto".
