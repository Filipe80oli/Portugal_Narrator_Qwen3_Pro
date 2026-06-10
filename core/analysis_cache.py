# core/analysis_cache.py
# ─── Guardar e carregar análises em JSON ──────────────────────────────────────

import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime

import tkinter as tk
import customtkinter as ctk

from config.settings import ANALYSIS_DIR

logger = logging.getLogger(__name__)


def compute_book_hash(text: str) -> str:
    """Hash MD5 de uma amostra do texto (para detetar alterações)."""
    sample = text[:50000] + text[-10000:] if len(text) > 60000 else text
    return hashlib.md5(sample.encode('utf-8', errors='ignore')).hexdigest()


def get_analysis_path(book_path: str) -> Path:
    """Retorna o caminho do ficheiro .analysis.json correspondente ao livro."""
    return ANALYSIS_DIR / f"{Path(book_path).stem}.analysis.json"


def save_analysis(book_path: str, raw_text: str, characters: dict,
                  segments: list, model_name: str) -> str | None:
    """Guarda a análise em JSON. Retorna o caminho ou None em caso de erro."""
    try:
        analysis_path = get_analysis_path(book_path)
        data = {
            "version": "1.0",
            "book_file": str(Path(book_path).resolve()),
            "book_hash": compute_book_hash(raw_text),
            "created_at": datetime.now().isoformat(),
            "ollama_model": model_name,
            "characters": _clean_for_json(characters),
            "segments": _clean_for_json(segments),
        }
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Análise guardada: {analysis_path.name}")
        return str(analysis_path)
    except Exception as e:
        logger.error(f"Erro ao guardar análise: {e}")
        return None


def load_analysis(analysis_path: str) -> dict | None:
    """Carrega análise de um JSON. Retorna o dicionário ou None em erro."""
    try:
        with open(analysis_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Erro ao carregar análise: {e}")
        return None


def _clean_for_json(obj):
    """Remove objetos não serializáveis (widgets tkinter)."""
    if isinstance(obj, dict):
        return {
            k: _clean_for_json(v) for k, v in obj.items()
            if not k.startswith('_') and not isinstance(v, (tk.Variable, ctk.Variable))
        }
    elif isinstance(obj, list):
        return [_clean_for_json(item) for item in obj]
    elif isinstance(obj, (tk.Variable, ctk.Variable)):
        return None
    return obj
