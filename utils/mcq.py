"""Estrazione robusta della risposta MCQ dall'output di un VLM/LLM.

Niente "regex secca": `find_mcq_answer` prova una batteria di strategie in
ordine di affidabilità e si ferma alla prima che dà un indice valido in
`[0, len(options))`. Restituisce anche *quale* strategia ha matchato, utile
per debug/osservabilità; `parse_mcq_letter` è il wrapper che ritorna il solo
indice (l'API stabile usata dagli scorer in `evals/`).

Strategie (in ordine):

1. ``leading``     — lettera ancorata a inizio output: "C", "(C)", "C.",
                     "[B]", "C) ...". Caso dominante quando il modello
                     obbedisce a "answer with the letter only". Richiede un
                     delimitatore di coda per non catturare l'iniziale di una
                     frase ("C" di "Clearly...").
2. ``cue``         — lettera isolata subito dopo un cue esplicito
                     ("the answer is B", "option C", "correct choice: D").
                     Si usa l'ULTIMO cue per saltare l'eco della domanda.
3. ``isolated``    — prima lettera A..E isolata (``\\bX\\b``) ovunque
                     nell'output, con guardia sull'articolo inglese "A".
4. ``option_text`` — fallback semantico: il modello ha scritto il
                     *contenuto* dell'opzione invece della lettera; si cerca
                     il testo normalizzato dell'opzione dentro l'output (a
                     parità, vince la più lunga / più specifica).
5. ``none``        — nessuna strategia ha matchato → ``(None, "none")``.

Eseguibile come script per ispezionare il comportamento su casi d'esempio:

    uv run python utils/mcq.py
"""
from __future__ import annotations

import re

# `\b` su input upper()-ato: matcha A..E come token isolato — pesca la
# lettera dentro "(A)", "A.", "Answer: B" ma non dentro "Atlanta" / "BEFORE".
# 5 lettere coprono tutti i benchmark MCQ cablati (max 5 candidates, MVBench).
_LETTER_RE = re.compile(r"\b([A-E])\b")

# Lettera ancorata a inizio output: "C", "C.", "(C)", "[B]", "C) ...".
# Il delimitatore di coda evita di catturare l'iniziale di una frase.
_LEADING_RE = re.compile(r"^\s*[\(\[]?\s*([A-E])\s*[\)\].:,]")

# Cue espliciti che in genere precedono la lettera nelle risposte verbose.
_CUE_RE = re.compile(r"\b(?:ANSWER|OPTION|CHOICE|CORRECT|LETTER)\b")

# Normalizzazione per il fallback semantico: minuscolo, via la punteggiatura,
# spazi collassati.
_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]+")
_WS_RE = re.compile(r"\s+")


def _norm(s: str) -> str:
    """Lowercase, punteggiatura→spazio, spazi collassati."""
    return _WS_RE.sub(" ", _NON_ALNUM_RE.sub(" ", s.lower())).strip()


def _letter_idx(letter: str, n: int) -> int | None:
    idx = ord(letter) - ord("A")
    return idx if 0 <= idx < n else None


def _scan_isolated(text_up: str, orig: str, n: int) -> int | None:
    """Prima lettera A..E isolata (``\\b…\\b``) valida in `text_up`,
    scartando l'articolo inglese "A" (cioè "A " seguito da parola minuscola
    nell'originale `orig`, che ha gli stessi indici di `text_up` perché
    `upper()` preserva la lunghezza)."""
    for m in _LETTER_RE.finditer(text_up):
        idx = _letter_idx(m.group(1), n)
        if idx is None:
            continue
        if m.group(1) == "A":
            tail = orig[m.end():m.end() + 2]
            if tail[:1] == " " and tail[1:2].islower():
                continue  # "A picture..." → articolo, non risposta
        return idx
    return None


def _match_option_text(text: str, options: list[str]) -> int | None:
    """Fallback semantico: ritorna l'indice dell'opzione il cui testo
    normalizzato compare nell'output; a parità, vince la più lunga (più
    specifica). None se nessuna match."""
    raw_n = _norm(text)
    if not raw_n:
        return None
    hits = [
        (len(opt_n), i)
        for i, opt in enumerate(options)
        if len(opt_n := _norm(opt)) >= 2 and opt_n in raw_n
    ]
    if not hits:
        return None
    hits.sort(reverse=True)
    return hits[0][1]


def find_mcq_answer(raw: str, options: list[str]) -> tuple[int | None, str]:
    """Mappa l'output del modello all'opzione MCQ scelta, provando le
    strategie a fallback in ordine.

    Ritorna `(idx, strategy)` dove `idx` è l'indice in `[0, len(options))`
    (o None) e `strategy` ∈ {"leading","cue","isolated","option_text","none"}
    indica quale strategia ha deciso — comodo per loggare/diagnosticare.
    """
    n = len(options)
    text = raw.strip()
    up = text.upper()

    m = _LEADING_RE.match(up)
    if m is not None:
        idx = _letter_idx(m.group(1), n)
        if idx is not None:
            return idx, "leading"

    last_cue = None
    for last_cue in _CUE_RE.finditer(up):
        pass
    if last_cue is not None:
        idx = _scan_isolated(up[last_cue.end():], text[last_cue.end():], n)
        if idx is not None:
            return idx, "cue"

    idx = _scan_isolated(up, text, n)
    if idx is not None:
        return idx, "isolated"

    idx = _match_option_text(text, options)
    if idx is not None:
        return idx, "option_text"

    return None, "none"


def parse_mcq_letter(raw: str, options: list[str]) -> int | None:
    """Indice dell'opzione MCQ scelta (o None). Wrapper su `find_mcq_answer`
    che scarta l'informazione sulla strategia. API stabile per gli scorer."""
    return find_mcq_answer(raw, options)[0]


# Casi d'esempio: documentazione eseguibile + smoke test del comportamento a
# fallback. `(raw, options, expected_idx, expected_strategy)`.
_DEMO_CASES: list[tuple[str, list[str], int | None, str]] = [
    # lettera nuda senza delimitatore → la pesca `isolated` (la regex
    # `leading` resta stretta apposta, per non catturare l'articolo "A").
    ("C", ["w", "x", "y", "z"], 2, "isolated"),
    ("(B)", ["w", "x", "y", "z"], 1, "leading"),
    ("D.", ["w", "x", "y", "z"], 3, "leading"),
    ("A", ["w", "x", "y", "z"], 0, "isolated"),
    ("The answer is B.", ["w", "x", "y", "z"], 1, "cue"),
    ("The correct option is clearly D, because...", ["w", "x", "y", "z"], 3, "cue"),
    ("A car is shown, the answer is C", ["w", "x", "y", "z"], 2, "cue"),
    ("a blue truck", ["a red car", "a blue truck", "a green bus", "a taxi"], 1, "option_text"),
    ("I think it's a green bus.", ["a red car", "a blue truck", "a green bus", "a taxi"], 2, "option_text"),
    ("Z", ["w", "x", "y", "z"], None, "none"),
    ("Mostly nonsense here.", ["w", "x", "y", "z"], None, "none"),
]


def _demo() -> int:
    ok = True
    for raw, options, exp_idx, exp_strat in _DEMO_CASES:
        idx, strat = find_mcq_answer(raw, options)
        passed = idx == exp_idx and strat == exp_strat
        ok = ok and passed
        print(
            f"{'OK ' if passed else 'FAIL'} idx={idx!r:>5} strat={strat:<11} "
            f"(atteso idx={exp_idx!r}, {exp_strat}) | {raw!r}"
        )
    print("\nTUTTI OK" if ok else "\n!!! CI SONO FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_demo())
