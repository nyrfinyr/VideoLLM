"""Preparazione della lista di sample per una eval: shuffle → shard → limit.

Estratto da `main.run()` per leggibilità: la logica (e il suo ordine) è
invariata, `run` chiama solo `prepare_samples`.
"""
import logging
import random

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def prepare_samples(samples: list[dict], cfg: DictConfig) -> list[dict]:
    """Applica nell'ordine shuffle → shard → limit e ritorna la lista risultante.
    """
    if cfg.get("shuffle"):
        random.Random(cfg.seed).shuffle(samples)
        logger.info("shuffle=true (seed=%d) → ordine sample randomizzato prima di limit", cfg.seed)

    num_shards = cfg.get("num_shards")
    if num_shards is not None:
        shard = cfg.get("shard")
        if shard is None:
            raise ValueError("num_shards impostato ma shard mancante")
        shard, num_shards = int(shard), int(num_shards)
        if not (0 <= shard < num_shards):
            raise ValueError(f"shard={shard} fuori range [0, {num_shards})")
        before = len(samples)
        samples = samples[shard::num_shards]
        logger.info("shard %d/%d → %d/%d sample (slice strided)", shard, num_shards, len(samples), before)
    limit = cfg.get("limit")
    if limit is not None:
        samples = samples[: int(limit)]
        logger.warning("limit=%s attivo → eval su %d sample (SUBSET, metriche di accuracy non significative)", limit, len(samples))
    return samples
