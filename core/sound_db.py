# core/sound_db.py
# ─── Base de dados de sons: índice, pesquisa fuzzy e metadados ────────────────
#
# Estrutura de pastas esperada:
#   sounds/
#     nature/rain.wav, nature/thunder.wav, ...
#     city/traffic.wav, ...
#     interior/door_open.wav, ...
#     action/explosion.wav, ...
#     ambience/tension.wav, ...
#     music/dramatic.wav, ...
#
# Se um som não existir na BD, a função retorna None (sem erros).

import json
import logging
from pathlib import Path
from difflib import get_close_matches

from config.settings import SOUNDS_DIR, SOUND_CATEGORIES

logger = logging.getLogger(__name__)

# ─── Índice em memória ────────────────────────────────────────────────────────
# Construído na primeira chamada a get_sound_path()
_INDEX: dict[str, Path] = {}   # "rain" -> Path("sounds/nature/rain.wav")
_INDEX_BUILT = False


def _build_index():
    global _INDEX, _INDEX_BUILT
    _INDEX.clear()
    if not SOUNDS_DIR.exists():
        _INDEX_BUILT = True
        return
    for wav in SOUNDS_DIR.rglob("*.wav"):
        key = wav.stem.lower()
        _INDEX[key] = wav
    # Também adicionar aliases das categorias
    for cat, keys in SOUND_CATEGORIES.items():
        for k in keys:
            if k not in _INDEX:
                # Tenta encontrar pelo nome exato numa subpasta
                candidate = SOUNDS_DIR / cat / f"{k}.wav"
                if candidate.exists():
                    _INDEX[k] = candidate
    _INDEX_BUILT = True
    logger.info(f"Sound DB: {len(_INDEX)} sons indexados em '{SOUNDS_DIR}'")


def get_sound_path(name: str) -> Path | None:
    """
    Devolve o caminho do ficheiro WAV mais próximo do nome pedido.
    Usa correspondência exata primeiro, depois fuzzy.
    Retorna None se não encontrar nada.
    """
    global _INDEX_BUILT
    if not _INDEX_BUILT:
        _build_index()

    key = name.lower().strip().replace(" ", "_")

    # Exato
    if key in _INDEX:
        return _INDEX[key]

    # Fuzzy (tolerância de 1-2 caracteres)
    matches = get_close_matches(key, _INDEX.keys(), n=1, cutoff=0.75)
    if matches:
        logger.debug(f"Sound DB fuzzy: '{key}' → '{matches[0]}'")
        return _INDEX[matches[0]]

    logger.debug(f"Sound DB: '{key}' não encontrado.")
    return None


def list_all_sounds() -> list[str]:
    """Lista todas as chaves disponíveis no índice."""
    if not _INDEX_BUILT:
        _build_index()
    return sorted(_INDEX.keys())


def rebuild_index():
    """Força reconstrução do índice (útil após adicionar sons à pasta)."""
    global _INDEX_BUILT
    _INDEX_BUILT = False
    _build_index()


def get_sounds_summary() -> dict:
    """Retorna um resumo por categoria para mostrar na UI."""
    if not _INDEX_BUILT:
        _build_index()
    summary = {}
    for cat in SOUND_CATEGORIES:
        cat_dir = SOUNDS_DIR / cat
        if cat_dir.exists():
            wavs = list(cat_dir.glob("*.wav"))
            summary[cat] = len(wavs)
        else:
            summary[cat] = 0
    summary["total"] = len(_INDEX)
    return summary
