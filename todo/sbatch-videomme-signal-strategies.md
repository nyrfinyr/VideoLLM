# Sbatch per valutare Video-MME con le nuove strategy signal-driven — IMPLEMENTATO (2026-07-14)

`scripts/sbatch/video_mme-signal-test.sbatch` (throughput probe / tuning
soglie, shuffle+limit=40) e `scripts/sbatch/video_mme-signal.sbatch` (prod,
2700 sample) — entrambi parametrizzati da `MODEL` (default
`qwen2_5_vl_3b_attn`) e `STRATEGY` (default `entropy_attention_resample`,
override `STRATEGY=entropy_shortcut`), `wandb.group`/`tags`/`name` includono
la strategy, `"$@"` inoltrato a `main.py` per override rapidi delle soglie
da riga di comando. Solo video-only per ora (niente variante subtitles).
Il prod NON va lanciato finché le soglie non sono validate con lo script di
test — vedi commento in testa a `video_mme-signal.sbatch`. Design/decisioni
sotto restano come riferimento.

**Tuning soglie (2026-07-14)**: `scripts/sweep_video_mme_thresholds.sh`
lancia lo sweep coarse richiesto sopra — riusa `video_mme-signal-test.sbatch`
via override CLI (nessun nuovo sbatch), grid 4×3 (`entropy_threshold` ∈
{0.3,0.5,0.7,1.0} × `concentration_threshold` ∈ {20,30,45}) + 1 run di
controllo `strategy=uniform`, tutti su `limit=250` con lo STESSO seed/shuffle
(subset identico, confronto mele-con-mele) — non i 40 del probe di
throughput (troppo pochi per un segnale di accuracy affidabile). `--time`
per i submit stimato (nessuna misura reale ancora), overridabile via
`SWEEP_TIME`/`CONTROL_TIME`; `DRY_RUN=1` per controllare i comandi prima di
sottomettere. Risultati confrontabili in Weave nel group
`<model>-sweep-video_mme-test`. `window_frac`/`phase_shift_frac`/
`sink_percentile` restano ai default in questo primo giro.

**Correzione post-review (2026-07-14)**: il costo di `entropy_attention_
resample` NON è "raddoppia solo sui sample ricampionati" — il pass 1
(`generate_with_signals`, prefill-only) gira SEMPRE prima della vera
`generate()`, quindi ogni sample paga un prefill extra a prescindere dal
ricampionamento (overhead grosso modo costante per-sample, non legato alla
frazione di resample). Il `--time` del prod è stato tolto (era un numero
indovinato, 18h) e sostituito con un placeholder volutamente rotto
(`00:01:00`, fail-fast) + istruzioni nel commento per calcolarlo dal
`model_latency.mean` misurato da `video_mme-signal-test.sbatch`, stesso
metodo di `docs/probe_throughput.md`. Constraint GPU lasciato invariato
(24G+45G, nessun pin a card veloci) su richiesta esplicita — rischio di
timeout su card lente accettato consapevolmente, non per svista.

`scripts/sbatch/video_mme.sbatch` (prod) e `video_mme-test.sbatch`
(throughput probe) lanciano solo `strategy=uniform` (default implicito,
mai passato esplicitamente). Da quando esistono `entropy_shortcut` e
`entropy_attention_resample` (`strategies/`, preset in `conf/config.yaml`
sotto `strategy:`), manca uno sbatch che le usi su Video-MME per
confrontarle con la baseline uniform già misurata.

## Da fare

- Nuovo `scripts/sbatch/video_mme-signal.sbatch` (o varianti per
  strategy), ricalcato su `video_mme.sbatch` con due differenze
  obbligatorie:
  - `model=qwen2_5_vl_3b_attn` (le strategy signal-driven richiedono
    `SupportsSignals` — fail-fast `RuntimeError` con `qwen2_5_vl_3b`
    plain, vedi `strategies/entropy_shortcut.py` /
    `entropy_attention_resample.py`).
  - `strategy=entropy_attention_resample` (quella con la logica a 4 rami
    già implementata — vedi `docs/entropia_confidenza_mcq.md` per il
    razionale) — eventualmente anche una variante `strategy=entropy_shortcut`
    per isolare l'effetto del solo skip-generate.
  - `wandb.group`/`wandb.tags`/`wandb.name` da aggiornare per includere
    `strategy` nel nome (oggi solo `${MODEL}-video_mme`), altrimenti i
    run signal-driven finiscono mescolati con la baseline uniform nello
    stesso group su wandb.
- Prima del run prod: uno smoke test tipo `video_mme-test.sbatch`
  (shuffle+limit) ma con le soglie di `entropy_attention_resample`
  (`entropy_threshold`, `concentration_threshold`, ecc. — vedi
  `conf/config.yaml`) NON ancora validate per Video-MME (erano tarate su
  un campione VNBench "retrieval", dominio/nframes diversi — nota già in
  `strategies/entropy_attention_resample.py`). Serve un giro di tuning
  soglie prima del run prod, non solo il porting dello sbatch.
- Video-MME ha `nframes=128` (contro es. 32 di MVBench, il dataset per
  cui la strategy è stata pensata inizialmente) — verificare che i tempi
  di resampling (finestra `window_frac`/`phase_shift_frac`) restino
  sensati a quella scala di frame/durata clip, non solo che il codice
  giri.

## Non ancora deciso

- Se il confronto va fatto su TUTTO Video-MME (2700 sample, costoso — vedi
  correzione sopra: il pass 1 extra è per-sample, non solo sui
  ricampionati) o solo su uno shard/subset rappresentativo prima di
  committare al run pieno.
- Se serve anche una variante con sottotitoli (`dataset.use_subtitles=true`)
  o solo la modalità video-only per questo primo confronto.
