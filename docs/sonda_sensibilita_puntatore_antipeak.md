# Sonda di sensibilità del canale-prompt (`cell_select=antipeak`)

> Perché esiste questo esperimento, cosa distingue esattamente, e come si
> legge il risultato. Riferimenti: arm `highlight_top1_24` /
> `random_top1_24` in `README.md` § "Ablation entropy/shift/zoom
> (Video-MME)", strategy in `strategies/attention_highlight.py`, sbatch in
> `scripts/sbatch/ablation-videomme/antipeak_probe_24.sbatch`.

## Esito (2026-07-29)

**Mondo (A): il canale-prompt è vivo.** Danno **6.46 pp**
(99.66% → 93.20% sui sample corretti al pass 1), z = 4.24, p < 0.0001 —
tabella completa e discussione in `README.md` § "Sonda di sensibilità del
canale-prompt (`antipeak`)". Il resto di questo documento è il razionale
con cui l'esperimento è stato disegnato, lasciato al tempo presente.

Conseguenza: il nulla di `highlight_top1_24` va attribuito al **segnale**
(picco d'attenzione posizionale, ~equivalente a una cella a caso), non al
canale. Le direzioni "entity rows" e "double-frame" restano aperte;
Qwen3-VL non serve più a stabilire se il modello ascolta.

## Il fatto di partenza

Tre numeri su Video-MME intero (2700 sample, `qwen2_5_vl_3b_attn`,
`nframes=24`):

| arm | accuracy pooled |
|---|---:|
| `baseline_24` (nessun puntatore) | 55.04% |
| `highlight_top1_24` (puntatore sul picco di attenzione) | 54.63% |
| `random_top1_24` (puntatore su una cella a caso) | 54.78% |

Aggiungere al prompt la frase *"Focus on the part of the video between X
and Y seconds: that is where the visual evidence relevant to the question
is"* non ha cambiato niente — **né quando X-Y venivano dall'attenzione, né
quando erano estratti a sorte**. Le tre righe stanno dentro mezzo punto.

## Le due letture possibili

Quel nulla ammette due spiegazioni incompatibili fra loro:

- **(A) Il puntatore funziona, ma punta nel posto sbagliato.** Il modello
  esegue l'istruzione; è il picco di attenzione a non essere dove sta
  l'evidenza. → la strada è migliorare il **segnale**: restringere le
  righe-query ai token dell'entità (`question_rows` in
  `utils/attn_core.py`, divergenza #2 dal paper LoT), raddoppiare i frame
  per portare la risoluzione del puntatore da 12 a 24 celle, o passare a
  Qwen3-VL per un grounding temporale più forte.
- **(B) Il modello non esegue l'istruzione.** La frase entra nel prompt e
  viene sostanzialmente ignorata. → **nessuna di quelle tre strade serve**,
  perché tutte migliorano *dove* si punta, e puntare meglio non produce
  nulla se l'indicazione non viene raccolta.

La differenza è operativa, non accademica: l'opzione "Qwen3-VL" richiede di
wirare da zero i segnali per una nuova famiglia di modelli
(`Qwen3VL2B`/`4B` sono stub). Sapere in quale dei due mondi siamo vale
quelle settimane.

## Perché i dati esistenti non lo dicono già

`random_top1_24` sembra il test naturale — indica un punto a caso, quindi
quasi sempre sbagliato, e non ha fatto danno. Ma è un test **senza
potenza**, per due motivi:

1. **Il gate.** Il puntatore scattava solo con `answer_entropy >
   entropy_threshold`, cioè sul ~73% dei sample "difficili", dove il
   modello sta al **42%** di accuracy. Sono sample in maggioranza *già
   sbagliati*: non c'è spazio per peggiorare. Un'istruzione dannosa
   applicata a chi sta già sbagliando non produce un segnale misurabile.
2. **La diluizione.** Con `t=12` celle e `top_k=1`, l'estrazione casuale
   pesca proprio la cella dell'attenzione ~1 volta su 12 (8.3%): su quei
   sample i due arm sono identici per costruzione.

Entrambi i run girarono con `always_highlight: false` ed
`entropy_threshold: 0.7` — verificato dalle config dei run wandb, non
dagli sbatch.

## Cosa fa la sonda

Cambia due cose, entrambe per massimizzare la sensibilità:

1. **Punta di proposito nel posto sbagliato.** `cell_select=antipeak`
   sceglie la cella più **lontana nel tempo** dal picco — non quella a
   massa minima, che su un profilo bimodale stretto può cadere accanto al
   picco.
2. **Legge il risultato sui soli sample già corretti al pass 1**, dove si
   parte da ~88% di accuracy e lo spazio per rompere qualcosa c'è tutto.
   È il motivo per cui `pred_pass1` va loggato: senza, quel sottoinsieme
   non è isolabile.

In una frase: *si prendono i sample che il modello risolve correttamente,
gli si dice con insistenza di guardare nel punto sbagliato del video, e si
guarda se si rompono.*

## Il punto chiave: la conclusione non dipende da quanto è buono il picco

È la ragione per cui questa sonda vale più di un esperimento con ground
truth temporale (che oltretutto su Video-MME non esiste — LVBench ce
l'ha via `time_reference`, ma è un regime da 768 frame su video di
un'ora, incomparabile con l'ablation a 24 frame):

- se il picco **è** informativo → l'antipicco è sbagliato → un modello che
  obbedisce peggiora;
- se il picco **non** è informativo (è recency posizionale — il probe da
  40 sample [`2dodrndg`](https://wandb.ai/alesvale97-unimore/video_mme/runs/2dodrndg)
  mostra il picco sull'ultima cella nel 28% dei casi, 3.4× il livello di
  caso) → l'antipicco è comunque un punto arbitrario → un modello che
  obbedisce peggiora **lo stesso**, su sample che stava già indovinando.

In entrambi i rami vale la stessa implicazione:

> **nessun danno ⟹ il puntatore non viene eseguito.**

Non serve sapere dov'è davvero l'evidenza per concludere.

## Perché servono due arm

`pred_pass1` è l'argmax della softmax ristretta alle lettere (pass 1);
`pred` è il parse di `generate()` (pass 2). Sono due percorsi di decoding
diversi, quindi una quota di sample "corretti al pass 1, sbagliati al pass
2" esiste **anche senza alcun puntatore**. Il secondo arm la misura.

| arm | config | cosa misura |
|---|---|---|
| `antipeak` | `cell_select=antipeak`, `always_highlight=true` | danno del puntatore sviante + discrepanza di decoding |
| `nopointer` | `always_highlight=false`, `entropy_threshold=999` | solo la discrepanza di decoding |

Nel secondo il gate non si apre mai, quindi `final_prompt = prompt` e
nessun puntatore viene emesso — ma i due forward avvengono lo stesso,
perché in `strategies/attention_highlight.py` la chiamata a `generate()`
sta **fuori** dall'`if gate_open`.

I due arm sono i due task di un job array sullo **stesso subset** (nessun
`shard=`, stesso `seed`/`shuffle`/`limit`), e condividono un pass 1
identico bit-per-bit — verificato nello smoke test: `answer_entropy` e
`peak_cell` coincidono fra i due.

## Come si legge

Metriche nel summary wandb, emesse da `mcq_accuracy` (`evals/base.py`) via
un breakdown **derivato** su `pred_pass1` — derivato e non passthrough
perché la strategy logga `pred_pass1` ma non può sapere se è giusta:
`answer` esiste solo dentro lo scorer.

```
danno = mcq_accuracy_correct_pass1_true[nopointer]
      - mcq_accuracy_correct_pass1_true[antipeak]
```

| esito | lettura | conseguenza |
|---|---|---|
| **danno ≈ 0** | mondo (B): canale-prompt morto su Qwen2.5-VL-3B | entity-rows e double-frame **chiusi**. Resta solo cambiare *meccanismo*: marker in-band su Qwen3-VL (nativo lì, impossibile su 2.5-VL — vedi docstring di `attention_highlight.py`) |
| **danno ≫ 0** | mondo (A): il modello obbedisce | il nulla di `highlight_top1_24` è colpa di *dove* si punta → double-frame (12 celle sono ~50 s su un long) ed entity-rows diventano investimenti sensati, con un tetto misurato contro cui confrontarli |
| **danno < 0** | il puntatore sviante *migliora* | non è una direzione: è un bug o un effetto di lunghezza/formato del prompt, da chiarire prima di tutto il resto |

Il complemento `mcq_accuracy_correct_pass1_false` (sample già sbagliati al
pass 1) dà la faccia opposta: quanto *recupera* un puntatore sviante.
Atteso ~0; se recupera, il puntatore agisce come rumore/regolarizzatore e
non come indicazione temporale — lettura che chiude comunque il filone.

### ⚠️ Controllare anche la numerosità, non solo la frazione

I sample con `pred_fallback=true` sono **esclusi** dal breakdown. Lì `pred`
è posto uguale a `pred_pass1` (il pass 2 non ha prodotto una lettera
parseabile e si ripiega sull'argmax del pass 1), quindi il sample non può
risultare danneggiato per costruzione: tenerlo dentro spingerebbe
`correct_pass1_true` verso 1.0, cioè verso "il puntatore non fa danno" —
proprio la conclusione che la sonda deve poter falsificare.

Non è un'ipotesi di scuola: nello smoke test con layer troncati il
fallback è scattato su 2/2 sample e avrebbe prodotto un falso "canale
morto". Sui run reali (`highlight_top1_24`, `random_top1_24`) il fallback
è 0/900, quindi l'esclusione non dovrebbe togliere praticamente nulla — ma
la validità della sonda non deve dipendere da quel fatto empirico.

Quindi: se `seen_correct_pass1_true` risulta molto sotto il numero di
sample corretti al pass 1, il puntatore sta degradando la **parsabilità**
dell'output invece della risposta. È un effetto diverso, da riportare a
parte e non da nascondere dentro un'accuracy.

## Cosa NON misura

Non misura se l'attenzione localizza bene, e non misura se un puntatore
*corretto* aiuterebbe. Misura una cosa sola, a monte di entrambe: **se il
modello è capace di ricevere un'indicazione temporale via prompt.** È la
domanda che decide quali esperimenti successivi hanno senso, e costa una
run.
