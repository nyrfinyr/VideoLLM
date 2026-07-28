"""Probe (grana grossa): il text-decoder di Qwen2.5-VL-3B sa estrarre il/i
SOGGETTO/I di una domanda MCQ, riportando le PAROLE ESATTE della domanda?
Solo-testo (nessun video). Poi mappiamo NOI le parole estratte alle righe
della mappa di attenzione (``query_tokens`` delle capture) via string-match:
se il modello copia verbatim, il mapping e' esatto e gratuito.

Prima versione (indici / substring parafrasato) archiviata in git history:
gli INDICI non funzionano (il 3B sbaglia il token ~2/3 delle volte), la
SUBSTRING semantica e' buona ma parafrasa. Qui: soggetto + vincolo verbatim
+ possibili multipli, e si MISURA il tasso di mapping riuscito.

Uso: ``uv run python scripts/probe_entity_extraction.py [n]``.
"""
import glob
import sys

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# ordine di priorita': prima Video-MME (il target), poi altri per allargare
CAPTURE_GLOBS = [
    "data/videomme_capture/*/capture.pt",
    "data/mvbench_obj_exist/*/capture.pt",
    "data/vnbench/output_v2/*/capture.pt",
    "data/vnbench/output_v3/retrieval/*/capture.pt",
]

SYS = (
    "You extract the SUBJECT(S) of a question: the entity or entities the "
    "question is about. There may be more than one subject. Report them using "
    "EXACTLY the words as they appear in the question (verbatim, the same "
    "words, no synonyms, no paraphrase, no extra words). Separate multiple "
    "subjects with commas. Output only the subject words."
)


def question_tokens(qt):
    out = []
    for row, _idx, _tok, text in qt:
        out.append((row, text))
        if "\n" in text:
            break
    return out


def load_questions(n):
    qs = []
    for g in CAPTURE_GLOBS:
        ds = g.split("/")[1]
        for p in sorted(glob.glob(g)):
            o = torch.load(p, map_location="cpu", weights_only=False)
            qr = question_tokens(o["query_tokens"])
            q_text = "".join(t for _, t in qr).strip()
            qs.append({"ds": ds, "stem": p.split("/")[-2], "qrows": qr, "text": q_text})
            if len(qs) >= n:
                return qs
    return qs


def map_phrase_to_rows(phrase, qrows):
    """Righe-token che coprono il primo match (case-insensitive) di `phrase`
    nel testo della domanda. [] se non trovato (=parafrasi/verbatim fallito)."""
    text = ""
    spans = []
    for row, t in qrows:
        start = len(text)
        text += t
        spans.append((start, len(text), row))
    ph = phrase.strip().strip(".,?!:;'\"").lower()
    if not ph:
        return []
    i = text.lower().find(ph)
    if i == -1:
        return []
    j = i + len(ph)
    return [row for (s, e, row) in spans if s < j and e > i]


def gen(model, processor, system, user, max_new=32):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [{"type": "text", "text": user}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    gen_ids = out[0][inputs.input_ids.shape[1]:]
    return processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    print("loading model (device_map=auto, offload CPU)...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    qs = load_questions(n)
    print(f"{len(qs)} domande\n" + "=" * 100, flush=True)

    tot_phrases = 0
    mapped_phrases = 0
    n_multi = 0
    for k, q in enumerate(qs):
        ans = gen(model, processor, SYS, f"Question: {q['text']}")
        phrases = [p.strip() for p in ans.split(",") if p.strip()]
        if len(phrases) > 1:
            n_multi += 1
        results = []
        for ph in phrases:
            rows = map_phrase_to_rows(ph, q["qrows"])
            tot_phrases += 1
            if rows:
                mapped_phrases += 1
            results.append((ph, rows))
        hit = all(r for _, r in results) and results
        flag = "OK " if hit else ("MISS" if results else "----")
        print(f"[{flag}] {q['ds']:16s} {q['text']}", flush=True)
        print(f"        -> raw={ans!r}", flush=True)
        for ph, rows in results:
            print(f"        -> {ph!r:28s} rows={rows if rows else 'NO MATCH (parafrasi)'}", flush=True)
    print("\n" + "=" * 100)
    print(f"SUMMARY: {len(qs)} domande | frasi estratte={tot_phrases} | "
          f"mappate verbatim={mapped_phrases}/{tot_phrases} "
          f"({100*mapped_phrases/max(tot_phrases,1):.0f}%) | domande multi-soggetto={n_multi}")
    print("DONE")


if __name__ == "__main__":
    main()
