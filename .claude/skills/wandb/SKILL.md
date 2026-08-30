---
name: wandb
description: Ispeziona qualsiasi run wandb di questo progetto via API (eval, probe, ablation, prefetch) — config, summary, metriche e check di correttezza per-sample da Weave. Usa quando l'utente chiede di verificare/controllare/leggere il risultato o l'esito di una run, di una probe, di un job SLURM, di uno sbatch o di un'ablation, o chiede accuracy/latency/confronto fra arm. NON usa SSH.
---

Le eval di questo repo girano su SLURM al cluster AImageLab. **Da questa
macchina non c'è accesso al cluster**: niente log, niente file di output.
L'unico canale sono le API di wandb (config + summary) e Weave (l'output
**per-sample** di ogni `predict`, dove finiscono i campi che le strategy
ritornano da `answer()`).

Il driver è `.claude/skills/wandb/driver.py` (path relativo alla root del
repo). Credenziali da `~/.netrc`, nessun login interattivo.

## Prima di tutto: chiedi il nome della run

**L'utente annota il nome di ogni run che lancia su HPC — chiediglielo.**
È la chiave affidabile per trovarla; `--list` è il ripiego se non se lo
ricorda.

```
Come si chiama la run? (es. qwen3_vl_2b_attn-marker_24_probe60-92821_0)
```

Il progetto **non** serve saperlo: se il nome non è nel progetto di default
il driver cerca in tutti quelli dell'entity.

## NON fare

- **Non tentare `ssh` al cluster.** `ailab02` è in `~/.ssh/config` ma la
  chiave non autentica da qui (`Permission denied (publickey)`, nessun
  ssh-agent). Tutto ciò che serve è su wandb.
- **Non fermarti allo stato del job.** `state=finished` NON è un check: un
  OOM per-sample viene catturato da Weave e lascia il job `finished` con zero
  predizioni valide. Il check è "`mcq_accuracy` esiste nel summary".
- **Non leggere l'accuracy per prima.** I check di correttezza vengono prima;
  se falliscono, l'accuracy misura l'esperimento sbagliato.

## Uso

```bash
# report completo di una run (eval o no: si riconosce da sé)
uv run python .claude/skills/wandb/driver.py <nome-run>

# se l'utente non ricorda il nome
uv run python .claude/skills/wandb/driver.py --list 15
uv run python .claude/skills/wandb/driver.py --list 5 --all-projects
uv run python .claude/skills/wandb/driver.py --projects

# confronto fra due arm: isola da solo i knob che differiscono
uv run python .claude/skills/wandb/driver.py --compare <run-a> <run-b>

# ritarare la walltime di un fullset da una probe
uv run python .claude/skills/wandb/driver.py <nome-run> --shard-size 900

# solo summary, senza Weave (secondi invece di ~1 min)
uv run python .claude/skills/wandb/driver.py <nome-run> --summary-only

# run con query_rows=entity: tabella domanda → entity estratta, un blocco
# per sample, per la verifica MANUALE (mostrala all'utente, non riassumerla)
uv run python .claude/skills/wandb/driver.py <nome-run> --entities
```

Entity di default letta da `conf/config.yaml` (`wandb.default.entity`).
Progetto di default `video_mme` per `--list`; per una run singola è
irrilevante (ricerca su tutti). Override: `--entity` / `--project`.

## Cosa produce

**Run di eval** (config con `model`+`dataset`) — check condizionali sui campi
che la strategy emette, quindi un arm che non li emette li salta invece di
fallire. Output reale su una probe:

```
✅ eval viva: mcq_accuracy=0.6000 presente nel summary (36/60)
✅ pred_fallback: 0/60 — tutte le risposte del pass 2 parseabili
✅ query_rows: n_query_rows < n_query_tokens su 60/60 (query_rows='question', in media il 26% delle righe tenute)
✅ sink_filter: il filtro cambierebbe la cella scelta sul 33% dei sample (20/60): sink_filter=false arriva davvero al ranking
✅ blocchi/timestamp: blocchi tutti pari e somma(block_sizes)/2 == t_cells su 60/60 sample marcati
   marked: 60/60 sample (100%)
   costo: model_latency_mean=12.13 s/sample, eval osservata 0.20 h su 60 sample | nframes=24
     walltime: a 900 sample/shard ≈ 3.03 h (margine su 4 h: 1.32×)
   accuracy: 0.6000 | sui corretti-al-pass-1: 0.892 (33/37)

VERDETTO: tutti i check di correttezza passati (5).
```

**Run non di eval** (prefetch, sweep, ...): niente sample né accuracy, si
stampano config e summary così come sono.

Per aggiungere un check a una strategy nuova: basta che il suo `answer()`
ritorni il campo, poi si aggiunge il ramo in `check_eval()`.

## Gotchas

- **Weave non registra da nessuna parte il run id di wandb.** Gli
  `attributes` dei call contengono solo versione python/OS. L'unico
  collegamento run→evaluation è la **vicinanza temporale**. Il driver non si
  fida di quell'euristica da sola: **verifica** l'accoppiamento confrontando i
  campi che la strategy ri-emette per sample (`cell_select`, `sink_filter`,
  `resample_kind`) con `config.strategy`. Serve davvero — due arm lanciati a
  10 secondi l'uno dall'altro (attention vs random) si distinguono solo così.
  Quando `config.strategy` è vuota (run vecchie, `uniform`) non c'è niente da
  confrontare e il driver **lo dichiara**: lì il match è solo temporale.
- **`parent_ids` non porta ai `predict`.** La gerarchia è
  `Evaluation.evaluate` → `Evaluation.predict_and_score` → `predict`, quindi
  il genitore di un `predict` non è l'evaluate. Si usa
  `trace_ids=[ev.trace_id]`, che scopa esattamente i sample di quella eval.
- **I check di correttezza NON sono nel summary.** `n_query_rows`,
  `block_sizes`, `peak_cell_sink_filtered` vivono solo nell'output per-sample
  di Weave. Il summary ha solo le chiavi di `BREAKDOWN_KEYS`
  (`evals/base.py`) — e `marked` non è fra quelle, quindi "quanti sample sono
  stati toccati dall'intervento" si sa solo da Weave.
- **Non giudicare un check su 2 sample.** Su una probe i primi due sample
  avevano `peak_cell == peak_cell_sink_filtered` e sembrava che il knob non
  arrivasse; sui 60 le celle differiscono nel 33% dei casi e il check passa.
  Il driver aggrega sempre su tutti i sample.
- **`--compare` non è un test statistico.** Su una probe da 60 un delta di 1
  sample è +1.7 pp, non distinguibile dal rumore; il driver lo stampa
  esplicitamente per non farlo leggere come un risultato.
- **La dimensione dello shard non è deducibile dalla run** (`limit` tronca, il
  totale del dataset non è nel config): l'estrapolazione della walltime esce
  solo con `--shard-size N`, e N viene sempre stampato.
- **`state=crashed` su un job lungo è spesso la walltime, non un bug** (già
  successo con l'array 76131). Il campo `costo` dà la latency misurata per
  ritarare `--time`. Esempio reale: `qwen2_5_vl_3b-lvbench-46824_2` risulta
  `finished` ma senza `mcq_accuracy`, a 83.5 s/sample — troncata dal tempo.
- Weave stampa a ogni avvio "version has been recalled" e le righe di login:
  rumore, non errori.

## Troubleshooting

| Sintomo | Causa / fix |
|---|---|
| `run '<nome>' non trovata in nessun progetto` | Nome sbagliato: richiedilo all'utente, o `--list`/`--projects` |
| `nessuna Evaluation weave vicina a questa run` | La run è morta prima di iniziare l'eval (crash al caricamento del modello): resta il summary |
| `accoppiamento non verificabile` | Più eval concorrenti nella stessa finestra con config indistinguibili: i check per-sample vengono saltati invece di riportare numeri di un'altra run |
| `mcq_accuracy ASSENTE dal summary` | Eval girata ma ogni sample ha sollevato (OOM di `full_visual_attention`), oppure job troncato dalla walltime. Il job resta `finished`: non fidarsi dello stato |
| ssh `Permission denied (publickey)` | Atteso: non usare ssh, vedi sopra |
