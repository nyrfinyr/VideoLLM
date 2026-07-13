"""Loader di configurazione leggero (rimpiazza Hydra).

`conf/config.yaml` è un FILE UNICO: un blocco `active:` sceglie quale
preset usare per ogni gruppo (`model`/`generation`/`dataset`/`strategy`/
`wandb`, ciascuno una chiave dentro il blocco omonimo, es.
`dataset.egoschema`, `dataset.mvbench`) più i knob top-level
(`seed`/`hf_home`/`limit`/`shuffle`/`shard`/`num_shards`). `load_config`
risolve `active` (override da CLI con `model=X`/`dataset=Y`/ecc., stessa
sintassi di prima), estrae il preset scelto per ogni gruppo, applica gli
override CLI restanti (`dataset.nframes=100`, `strategy.entropy_threshold=1.0`
— questi si applicano al preset GIÀ RISOLTO, non alla struttura grezza),
poi risolve le interpolazioni. Composizione esplicita in Python, non un
motore implicito.

Interpolazioni supportate (le uniche due forme usate nei file YAML di
questo repo): `${oc.env:VAR}` / `${oc.env:VAR,default}` (variabile
d'ambiente) e `${percorso.punteggiato}` (riferimento a un'altra chiave
già risolta nell'albero finale, es. `${dataset.name}` nel preset
`wandb.default`). Risolte in UN solo passaggio, alla fine della
composizione — nessuna interpolazione presente nel repo referenzia
un'altra interpolazione ancora non risolta, quindi non serve un
fixed-point iterativo.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

CONF_DIR = Path(__file__).resolve().parent.parent / "conf"

# Gruppi selezionabili da CLI come `model=X`, `dataset=Y`, ecc. Un token
# CLI il cui path punteggiato è ESATTAMENTE uno di questi nomi seleziona
# quale preset usare per il gruppo (chiave dentro `conf/config.yaml`'s
# `<gruppo>: {<nome>: {...}}`), invece di essere un override di valore
# dentro un gruppo già risolto.
GROUPS = ("model", "generation", "dataset", "strategy", "wandb")

_INTERP_FULL_RE = re.compile(r"^\$\{([^}]+)\}$")
_INTERP_ANY_RE = re.compile(r"\$\{([^}]+)\}")


class Cfg(dict):
    """Dict con accesso ad attributo (`cfg.model.name`), oltre a `.get(...)`
    nativo (essendo una sottoclasse di `dict`).

    I valori nested restano dict PIATTI in memoria — il wrapping in `Cfg`
    avviene solo all'accesso via attributo (`__getattr__`), mai alla
    costruzione: così `dict(cfg)` resta sempre uno shallow-copy
    direttamente serializzabile in YAML/JSON, senza bisogno di una
    conversione ricorsiva dedicata.
    """

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Cfg(value) if isinstance(value, dict) else value


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _dotted_set(d: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def parse_overrides(argv: list[str]) -> list[tuple[str, Any]]:
    """`["model=qwen3_vl_4b", "dataset.nframes=100", "wandb.tags=[a,b]"]`
    → `[("model", "qwen3_vl_4b"), ("dataset.nframes", 100), ("wandb.tags", ["a", "b"])]`.

    Il RHS passa da `yaml.safe_load` per la type coercion (liste `[a,b]`,
    bool, `null`, numeri, stringhe nude) — stessa sintassi CLI già in uso
    negli sbatch/fetch script di questo repo.
    """
    overrides = []
    for token in argv:
        if "=" not in token:
            raise ValueError(f"override non valido (atteso key=value): {token!r}")
        path, _, raw_value = token.partition("=")
        overrides.append((path, yaml.safe_load(raw_value)))
    return overrides


def _resolve_expr(expr: str, root: dict) -> Any:
    if expr.startswith("oc.env:"):
        var, _, default = expr[len("oc.env:"):].partition(",")
        if var in os.environ:
            return os.environ[var]
        return yaml.safe_load(default) if default else None
    cur: Any = root
    for part in expr.split("."):
        cur = cur[part]
    return cur


def _resolve(value: Any, root: dict) -> Any:
    """Risolve ricorsivamente `${...}` su tutto l'albero. Una stringa che è
    ESATTAMENTE un'interpolazione ritorna il tipo nativo del valore
    risolto (es. `null`/int); un'interpolazione embedded in una stringa
    più ampia (es. `${oc.env:HOME}/datasets/x`) fa sostituzione testuale.
    """
    if isinstance(value, dict):
        return {k: _resolve(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, root) for v in value]
    if not isinstance(value, str) or "${" not in value:
        return value
    m = _INTERP_FULL_RE.match(value)
    if m:
        return _resolve_expr(m.group(1), root)
    return _INTERP_ANY_RE.sub(lambda m: str(_resolve_expr(m.group(1), root)), value)


def _check_required(value: Any, path: str = "") -> None:
    """Solleva `ValueError` su qualunque foglia `???` (stesso marker di
    "valore obbligatorio" usato nei `conf/*.yaml` esistenti, es.
    `videos_root: ???` in `conf/fetch/prefetch_vnbench.yaml`) rimasta
    non sovrascritta da un override CLI — fail-fast esplicito invece di
    propagare la stringa letterale `"???"` come path/valore.
    """
    if isinstance(value, dict):
        for k, v in value.items():
            _check_required(v, f"{path}.{k}" if path else k)
    elif value == "???":
        raise ValueError(f"valore obbligatorio mancante per {path!r} (era '???' — passalo da CLI: {path}=...)")


def load_config(argv: list[str], conf_dir: Path = CONF_DIR) -> Cfg:
    """Compone `conf/config.yaml` (file unico) + override CLI, in quest'ordine:

    1. `raw` = `<conf_dir>/config.yaml` (blocco `active:` + un blocco per
       gruppo con tutti i preset disponibili + knob top-level).
    2. Per ogni gruppo: preset da CLI se presente (`model=X`), altrimenti
       il default di `raw["active"]`; estrae `raw[gruppo][preset]` come
       contenuto risolto del gruppo (`KeyError` → messaggio con i preset
       validi).
    3. Applica gli override CLI restanti (non selettori di gruppo) via
       dotted-path-set sull'albero RISOLTO (funziona sia su chiavi
       top-level tipo `limit=5` sia annidate tipo `dataset.nframes=100`).
    4. Risolve le interpolazioni sull'albero finale.

    `conf_dir` di default punta a `conf/` del repo; parametrizzabile per
    test.
    """
    raw = _load_yaml(conf_dir / "config.yaml")
    active: dict = dict(raw.get("active", {}))
    overrides = parse_overrides(argv)

    group_selectors = {path: value for path, value in overrides if path in GROUPS}
    value_overrides = [(path, value) for path, value in overrides if path not in GROUPS]
    active.update(group_selectors)

    cfg: dict = {}
    for group in GROUPS:
        choice = active.get(group)
        if choice is None:
            continue
        try:
            cfg[group] = dict(raw[group][choice])
        except KeyError:
            valid = sorted(raw.get(group, {}).keys())
            raise ValueError(f"{group}={choice!r} non trovato in conf/config.yaml. Validi: {valid}") from None

    for key, value in raw.items():
        if key not in GROUPS and key != "active":
            cfg[key] = value

    for path, value in value_overrides:
        _dotted_set(cfg, path, value)

    resolved = _resolve(cfg, cfg)
    _check_required(resolved)
    return Cfg(resolved)


def load_flat_config(path: Path, argv: list[str]) -> Cfg:
    """Per script standalone con un unico YAML flat (es. `fetch/prefetch_*.py`,
    che non compongono più gruppi ma hanno un blocco `wandb:` annidato
    dentro lo stesso file). Stessa logica di override/interpolazione di
    `load_config`, senza selezione di gruppo.
    """
    cfg = _load_yaml(path)
    for p, value in parse_overrides(argv):
        _dotted_set(cfg, p, value)
    resolved = _resolve(cfg, cfg)
    _check_required(resolved)
    return Cfg(resolved)
