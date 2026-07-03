# Benchmark per il grounding video — riferimento per la tesi (LoT → KB-VideoQA)

> **Scopo del documento.** Mappare i benchmark accreditati utili per validare il transfer del framework *Look Twice* (LoT) al dominio video, organizzati secondo la framing **tri-axis**: **temporale × spaziale × retrieved passages**.
>
> **Distinzione di ruolo (importante).** I dataset qui elencati servono a *strumentare e validare la fedeltà dell'evidenza* (le mappe di cross-attention localizzano davvero l'entità/momento rilevante?), **non** a sostituire il benchmark finale knowledge-based. Il target del contributo resta **Video SimpleQA** (asse "retrieved passages" + KB), che nessuno di questi copre.

---

## Legenda

- **Asse**: T = temporale, S = spaziale, S-T = spazio-temporale.
- **GT**: tipo di ground-truth disponibile per il grounding.
- **Ruolo tesi**:
  - *Sanity-check* = verifica qualitativa/quantitativa che l'attenzione sia estraibile e ben localizzata.
  - *Faithfulness* = misura quantitativa che l'evidenza supporta la risposta (asse del contributo).
  - *Strumentale* = standard accreditato per validare il modulo di grounding, non il task finale.

---

## 1. Temporal grounding — moment retrieval / temporal sentence grounding

Task: data una query in linguaggio naturale, localizzare il segmento temporale (start/end). Standard "storici", GT temporale abbondante, metriche consolidate.

| Dataset | Asse | Task | GT | Durata clip | N. esempi | Metrica principale | Ruolo tesi | Note / limiti |
|---|---|---|---|---|---|---|---|---|
| **Charades-STA** | T | Query → segmento | Timestamp (start/end) | Brevi, indoor (~30s) | ~12,4k train / ~3,7k test | R@1 @IoU{0.5,0.7}, mIoU | Strumentale | Dominio ristretto (attività domestiche) |
| **ActivityNet Captions** | T | Query → segmento (eventi densi) | Timestamp multipli | Lunghi, non tagliati (~120s) | ~20k video, ~3,65 eventi/video | R@1 @IoU{0.3,0.5,0.7}, mIoU | Strumentale | Dominio ampio; bias temporali noti |
| **QVHighlights** | T | Moment retrieval + highlight detection | Timestamp + saliency clip-wise | 150s fissi | 10.148 clip, 10.310 query, 18.367 highlight | R@1, mAP, HIT@1 | Strumentale | Multi-moment; server di valutazione online sul test |
| **TACoS** | T | Query → segmento (fine-grained) | Timestamp | Lunghi, cucina | ~127 video, ~18k frasi | R@1 @IoU, mIoU | Opzionale | Molto fine-grained, dominio cucina |
| **DiDeMo** | T | Distinct moments | Segmenti discretizzati (5s) | ~25-30s | ~10k video, ~40k descrizioni | R@1, R@5, mIoU | Opzionale | Granularità temporale grezza |
| **YouCook2** | T | Dense captioning grounded | Timestamp + caption | Lunghi, cucina | ~2k video | CIDEr/METEOR + IoU, F1 | Opzionale | Più captioning che retrieval puro |

## 2. Temporal grounding — QA-grounded (più allineati alla tesi)

Task: il grounding temporale è legato al *rispondere a una domanda*. Misura direttamente la faithfulness dell'evidenza → asse temporale del contributo.

| Dataset | Asse | Task | GT | Durata clip | N. esempi | Metrica principale | Ruolo tesi | Note / limiti |
|---|---|---|---|---|---|---|---|---|
| **NExT-GQA** | T | Grounded VideoQA (MCQ) | ~10,5k timestamp legati alle domande | Brevi (~40s) | val + test su NExT-QA | **Acc@GQA** (risposta corretta ∧ IoU≥0.5), Acc@QA, mIoU/IoP | **Faithfulness** (standard de facto) | Clip brevi, poca diversità di dominio |
| **ReXTime** | T | Reasoning temporale cross-event grounded | Timestamp | Medi (ActivityNet/QVH) | ~migliaia di QA | Acc@QA, mIoU | Faithfulness | Spesso valutato in coppia con NExT-GQA |
| **CG-Bench** | T | Clue-grounded QA su video lunghi | Segmenti "clue" + MCQ | Lunghi (≈ ora) | ~1,2k video, ~12k QA | Acc + clue-grounded (white/black-box) | Faithfulness (caso long-video) | Nasce per superare i limiti di NExT-GQA |
| **DeVE-QA** (2025) | T | Grounded event QA | Timestamp eventi | Vari | — *(verificare sul paper)* | Acc@GQA-like | Faithfulness (recente) | Stat. da confermare dalla fonte |

## 3. Spatial / spatio-temporal grounding (asse spaziale — novelty)

Task: localizzare l'entità nello spazio (bounding box / tube), eventualmente nel tempo. Necessari per validare l'attenzione **spaziale** di LoT contro GT a box.

| Dataset | Asse | Task | GT | Durata clip | N. esempi | Metrica principale | Ruolo tesi | Note / limiti |
|---|---|---|---|---|---|---|---|---|
| **VidSTG** | S-T | STVG (frasi dichiarative + interrogative) | Tube di bounding box | Medi, non tagliati | 99.943 frasi, ~10,3k video, 80 categorie oggetti | m_vIoU, vIoU@R, m_tIoU, m_sIoU | **Faithfulness** (spaziale) | Lo split *interrogativo* è QA-like; è referring, non KB |
| **HC-STVG** | S-T | STVG human-centric | Tube box (1 frase/persona) | Medi, multi-persona | 5.660 video (v1: 4.5k/1.16k) | vIoU@R | Faithfulness (entità-persona) | Solo persone; utile per le tue domande tipo "man in suit" |
| **Perception Test — Grounded VideoQA** | S-T | QA + tracking entità | Tracklet di bounding box | ~23s media | train 586 vid/1.859 Q; val 1.545 vid/3.051 Q | IoU-based / mAP | Faithfulness (QA + spaziale insieme) | Unisce risposta e box; ottimo per il test integrato |
| **OmniGround** (2025) | S-T | Open-vocabulary STVG | Tube box, alta complessità | Vari | 3.475 video, 81 categorie | vIoU | Strumentale (open-vocab) | Più di nicchia; pipeline annotazione robusta a occlusioni |
| **ViTXT-GQA** | S-T | Grounded scene-text VideoQA | 52k box scene-text + segmenti | Vari | 729 video, 2k domande, 2,2k segmenti | vIoU + Acc | Opzionale | Solo se servono casi scene-text/OCR |

## 4. Benchmark di contesto (percezione e target finale)

Non forniscono GT di grounding, ma servono come riferimento.

| Dataset | Tipo | Ruolo tesi | Note |
|---|---|---|---|
| **MVBench** | Perception VideoQA (20 task, clip brevi) | Sanity-check (qualitativo) | Entità localizzate in pochi frame; nessun GT di grounding |
| **Video-MME** | Comprehensive VideoQA (12 task type) | Contesto / confronto competitor | Eterogeneo (short/medium/long); attenzione a v1 vs v2 |
| **Video SimpleQA** | Knowledge-based VideoQA | **Target del contributo** | Asse "retrieved passages" + KB; benchmark principale della tesi |

---

## Note metodologiche

1. **Separare i ruoli nel framing document.** Tenere una sezione *"Evaluation of evidence faithfulness"* (Tabelle 2 e 3) distinta dalla sezione *"Main benchmark"* (Video SimpleQA). Presentare risultati su VidSTG/NExT-GQA come fossero il contributo è un rischio: sono strumentazione, non il task KB.

2. **Asse temporale.** Per il *meccanismo* → Charades-STA / QVHighlights (GT abbondante, metriche accettate). Per il *contributo* → NExT-GQA (standard de facto), CG-Bench per il caso long-video.

3. **Asse spaziale (novelty vs. VideoSpeculateRAG).** VidSTG per la copertura generale, HC-STVG per le entità-persona, Perception Test per QA + box nello stesso dataset.

4. **Coerenza con i competitor.** Verificare quale *versione* di Video-MME (v1 vs v2, non confrontabili) e quali metriche usano i competitor prima di riportare numeri comparativi.

5. **Verificare le cifre marcate.** Le voci con `—` o `~` (in particolare DeVE-QA, CG-Bench) vanno confermate sui paper originali prima dell'inserimento definitivo.

---

## Riferimenti (paper originali)

- Charades-STA — Gao et al., *TALL: Temporal Activity Localization via Language Query*, ICCV 2017.
- ActivityNet Captions — Krishna et al., *Dense-Captioning Events in Videos*, ICCV 2017.
- QVHighlights — Lei et al., *Detecting Moments and Highlights in Videos via Natural Language Queries*, NeurIPS 2021.
- TACoS — Regneri et al., 2013. · DiDeMo — Hendricks et al., ICCV 2017. · YouCook2 — Zhou et al., 2018.
- NExT-GQA — Xiao et al., *Can I Trust Your Answer? Visually Grounded Video Question Answering*, CVPR 2024.
- ReXTime — Chen et al., 2024. · CG-Bench — Chen et al., 2024 (arXiv:2412.12075).
- VidSTG — Zhang et al., *Where Does It Exist: Spatio-Temporal Video Grounding for Multi-Form Sentences*, CVPR 2020.
- HC-STVG — Tang et al., 2021. · OmniGround / OV-STVG — Gao et al., 2025.
- Perception Test — Pătrăucean et al., NeurIPS 2023. · ViTXT-GQA — Zhou et al., 2024.
- Video-MME — Fu et al., 2024 (arXiv:2405.21075); Video-MME-v2 — 2026 (arXiv:2604.05015).
