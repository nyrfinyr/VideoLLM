# Oracolo su LVBench: dove sta davvero il guadagno

> Perché questo esperimento esiste, cosa separa, e come si legge. Sonde:
> `scripts/probe_peak_in_window.py`, `scripts/probe_oracle_ceiling.py`
> (sbatch omonimi in `scripts/sbatch/`). Dati: gruppi wandb
> [`probe-peak-in-window`](https://wandb.ai/alesvale97-unimore/lvbench/groups/probe-peak-in-window)
> (job 93613) e [`probe-oracle-ceiling`](https://wandb.ai/alesvale97-unimore/lvbench/groups/probe-oracle-ceiling)
> (job 93614 a tre condizioni, **94348 a quattro — è quello da citare**).

## Esito (2026-08-30)

**Marcare la cella giusta non paga mai; ricampionare dentro la finestra
giusta paga +16 pp.** Su 100 sample LVBench, `qwen3_vl_2b_attn`, 24 frame:

| condizione | accuracy | Δ vs baseline | appaiato | p (McNemar esatto) |
|---|---:|---:|---:|---:|
| `baseline` — 24 frame uniformi, nessun intervento | 32% | — | — | — |
| `peak` — puntatore testuale alla cella di picco | 29% | −3 | +4/−7 | 0.55 |
| `oracle` — **stesso puntatore** alla finestra ANNOTATA | 30% | −2 | +5/−7 | 0.77 |
| **`zoom`** — 24 frame ricampionati DENTRO la finestra annotata | **48%** | **+16** | **+22/−6** | **0.0037** |

Le due metà di LoT si separano nettamente, e in direzioni opposte:

- **la metà "marcare" (COME comunicare la salienza) è morta per
  principio.** Il soffitto non esiste: un puntatore perfetto vale quanto
  nessun puntatore. I tre null di Video-MME (`attention_highlight`,
  `attention_marker`, `visual_prompt` — vedi `260901-tabelle-risultati.md`)
  non erano un problema di segnale né di canale: il canale non ha niente da
  dare. Con `antipeak` sappiamo già che il modello **legge** l'istruzione
  (−6.46 pp, `sonda_sensibilita_puntatore_antipeak.md`); ora sappiamo che
  eseguirla correttamente non serve.
- **la metà "ricampionare" (DOVE guardare) ha un soffitto grande**, e il
  collo di bottiglia si sposta sulla localizzazione della finestra.

Il confronto diretto fra le due metà, appaiato sugli stessi sample, è
`zoom` contro `oracle`: **+23/−5, p = 0.0009**. Stessa informazione
d'oracolo, consegnata in due modi; solo uno funziona.

## Il meccanismo, non solo il numero

Il guadagno di `zoom` non è un effetto diffuso: è concentrato esattamente
dove la teoria lo prevede.

| | n | baseline | zoom |
|---|---:|---:|---:|
| finestra con **< 1** frame atteso nel campione uniforme | 77 | 27% | **49%** |
| finestra con **≥ 1** frame atteso | 23 | 48% | 43% |

- **Tutti e 22 i sample recuperati da `zoom` stanno nel primo gruppo.** Dove
  il campione uniforme già colpiva l'evidenza, ricampionare non aggiunge
  nulla (anzi, −5 pp: si perde il contesto del resto del video).
- Il regime lo spiega: su LVBench la durata mediana è **71 minuti** e la
  finestra annotata mediana **30 secondi**, cioè lo **0.8%** del video. Con
  24 frame uniformi il numero atteso di frame dentro la finestra è **0.20**
  (mediana); in **77 casi su 100 è sotto 1**.

Da qui la lettura che conta: **il campionamento uniforme non è "cieco"
perché sceglie male i frame che mostra, ma perché l'evidenza non è tra i
frame che mostra affatto.** Un puntatore testuale può solo dire al modello
di guardare meglio pixel che non contengono la risposta — ed è il motivo
per cui tre canali di consegna diversi hanno dato lo stesso nulla.

## Il segnale d'attenzione, misurato in modo diretto

La sonda `probe_peak_in_window` chiede al segnale se punta al posto giusto
**senza passare dall'accuracy**: hit = la cella di picco interseca la
finestra annotata. Il livello di caso è calcolato per-sample (finestre
larghe coprono più celle), non è `1/t`.

| rowset | hit | | 60 sample (93613) |
|---|---:|---|---|
| `all` — tutte le righe query | 28.3% (17/60) | | chance media **15.8%** |
| `question` — solo la domanda | 28.3% (17/60) | | appaiato entity/question: |
| `entity` — soggetto estratto | 30.0% (18/60) | | both 15, solo-entity 3, solo-question 2 |

Confermato in modo indipendente sul campione da 100 della sonda del
soffitto: **31/100 contro 18.8 attesi, z = 4.21**.

- **Il segnale esiste**, circa 2× il caso: il picco d'attenzione non è
  rumore puro (e questo aggiorna la lettura di `antipeak`, dove il picco
  sembrava "~equivalente a una cella a caso" — misurato *attraverso*
  l'accuracy, che qui sappiamo essere lo strumento sbagliato).
- **Ma è debole, e affilare le righe-query non lo migliora.** I tre rowset
  stanno dentro 1.7 pp, con 3 recuperi contro 2 rotture nel confronto
  appaiato. L'estrazione dell'entità (`query_rows=entity`, v4: una
  sequenza contigua verbatim o `NONE`) è meccanicamente sana anche su
  LVBench — 65% di frasi mappate, un solo `NONE` — quindi il limite non è
  l'implementazione: è che la posizione del picco non dipende granché da
  quali righe della domanda si mediano.
- **E soprattutto: è debole proprio dove servirebbe.** Sui 22 sample in cui
  `zoom` recupera la risposta — cioè dove sta tutto il guadagno — il picco
  cade nella finestra solo **4 volte (18%)**. Un arm "ricampiona dove punta
  il picco" ne catturerebbe una piccola frazione.

## Conseguenze

1. **Chiuso**: gli arm di marcatura/puntamento. Il negativo è forte perché
   ha un controllo d'oracolo, non solo un controllo random: non "il nostro
   segnale non basta", ma "nessun segnale basterebbe, su questo canale". Da
   qui la decisione di **non lanciare il fullset `visual_prompt` su
   Video-MME** (3 shard × ~3 h, con o senza `query_rows=entity`): avrebbe
   misurato un arm della metà chiusa — registro delle probe esistenti in
   `260901-tabelle-risultati.md` §5.
2. **Aperto e riformulato**: il ricampionamento guidato. La domanda di
   ricerca non è più *come comunicare la salienza* ma **quale segnale
   localizza la finestra d'evidenza** abbastanza bene da guidare un secondo
   campionamento.
3. **Il prossimo esperimento** è la quinta condizione della stessa sonda:
   `zoom` sulla cella di **picco** invece che sulla finestra vera — l'arm
   realistico, che misura quanto dei 16 pp il segnale attuale cattura
   davvero (+1 forward per sample). Vale la pena provare nella stessa corsa
   le varianti tolleranti alla localizzazione imprecisa (top-2 celle,
   finestra allargata), che con un segnale al 31% possono valere più del
   top-1 secco.

## Note di metodo

- **Un solo forward di cattura per sample**, poi il picco viene ricalcolato
  offline per i tre rowset sulla stessa attenzione: i confronti fra rowset
  sono appaiati per costruzione, non affetti da varianza di campionamento.
- **Condizioni appaiate**: stessi sample, stesso prompt MCQ, e per
  `peak`/`oracle` **lo stesso identico testo** di puntatore dell'arm
  `attention_highlight` (`POINTER_SPAN_SEC`, prepend). Fra le tre cambia
  solo *dove* punta, mai *come* parla.
- **`zoom` non bara sui timestamp**: i 24 frame ricampionati passano con
  `frames_indices`/`fps` reali, quindi il processor Qwen3 emette i
  timestamp assoluti veri del punto del video da cui vengono.
- **Quantizzazione a celle** anche per l'oracolo: la finestra annotata è
  arrotondata ai confini di cella (2 frame = 1 cella), che è la risoluzione
  vera del canale. Un puntatore perfetto non può dire di più.
- **Le due sonde condividono i sample** a parità di seed (selezione
  deterministica), e `peak_in_window_rate` è ricalcolato in entrambe come
  cross-check: 30-31% in due implementazioni indipendenti.
- **Precisione**: 100 sample danno un SE di ~4.7 pp su una singola
  accuracy, ma i test appaiati (McNemar) sono molto più sensibili — ed è
  con quelli che `zoom` passa e `oracle` no.
- Un primo giro (93610/93611) è morto con
  `'Qwen3VLTextAttention' object has no attribute '_last_attn'`: le sonde
  costruivano il modello a mano invece che dal preset
  `qwen3_vl_2b_attn`, che è ciò che attiva
  `attn_implementation.text_config=qwen_attn_capture`. Corretto; le sonde
  ora leggono i kwargs dal preset come fa `main.py`.
