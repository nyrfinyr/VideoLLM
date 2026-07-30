"""Payload metriche per la notifica Telegram di fine job.

Questo modulo NON parla con Telegram: scrive un file di testo che il trap di
uscita di `scripts/lib/notify.sh` allega al messaggio del job. La divisione è
voluta — il messaggio deve partire ANCHE quando Python non arriva in fondo (OOM,
timeout SLURM, `scancel`), e in quel caso il file resta vuoto e il messaggio
riporta solo l'esito.

Perché il dump è generico: `utils.obs.log_eval_summary` scrive già ogni numero
dell'eval in `wandb.run.summary` (accuracy totale, `n_correct`/`n_samples`,
l'intero breakdown per categoria di `breakdown_rows`, la latenza media). Qui si
rilegge quel dizionario per intero invece di elencare le chiavi: una metrica
aggiunta in futuro a `wandb.run.summary` compare su Telegram da sola, senza
toccare né questo file né gli sbatch.
"""
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Un messaggio Telegram sta in 4096 caratteri; il resto del messaggio (header,
# esito, nodo, eventuale coda dello stderr) sta abbondantemente nei ~600 di
# margine lasciati qui.
_MAX_CHARS = 3500
_MAX_VALUE_CHARS = 60


def _fmt(value) -> str:
    if isinstance(value, bool) or isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    text = str(value).replace("\n", " ")
    return text if len(text) <= _MAX_VALUE_CHARS else text[: _MAX_VALUE_CHARS - 1] + "…"


def _render() -> str | None:
    import wandb

    run = wandb.run
    if run is None:  # wandb.mode=disabled: niente da riportare
        return None

    # Le chiavi `_`-prefissate sono interne a wandb (`_runtime`, `_timestamp`,
    # `_step`, `_wandb`) e non sono metriche dell'esperimento.
    items = sorted((k, v) for k, v in dict(run.summary).items() if not k.startswith("_"))

    lines = [line for line in (run.name, run.url) if line]
    if lines and items:
        lines.append("")
    width = max((len(k) for k, _ in items), default=0)
    lines += [f"{k:<{width}}  {_fmt(v)}" for k, v in items]

    text = "\n".join(lines)
    return text if len(text) <= _MAX_CHARS else text[:_MAX_CHARS] + "\n… (troncato)"


def dump_run_summary() -> None:
    """Scrive in `$TG_SUMMARY_FILE` tutte le metriche di `wandb.run.summary`.

    No-op se la variabile non è nell'ambiente (job non lanciato tramite
    `scripts/lib/notify.sh`, o bot non configurato). Non solleva mai: una
    notifica rotta non deve far fallire un eval da ore, quindi ogni errore
    diventa un warning nel log del job.
    """
    path = os.environ.get("TG_SUMMARY_FILE")
    if not path:
        return
    try:
        text = _render()
        if text:
            Path(path).write_text(text)
    except Exception:  # noqa: BLE001 — vedi docstring
        logger.warning("Telegram: dump del summary fallito", exc_info=True)
