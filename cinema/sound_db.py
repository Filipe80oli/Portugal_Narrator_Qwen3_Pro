# cinema/sound_db.py
# ─── Base de dados de sons: indexação e pesquisa por nome ─────────────────────

import logging
from pathlib import Path
from difflib import SequenceMatcher

from config.settings import SOUNDS_DIR, SOUND_CATEGORIES

logger = logging.getLogger(__name__)


class SoundDB:
    """
    Indexa todos os ficheiros WAV/MP3 dentro de SOUNDS_DIR e permite
    encontrar o ficheiro mais próximo de um nome/keyword solicitado.

    Estrutura esperada (mas não obrigatória):
        sounds/
          nature/rain_heavy.wav
          nature/thunder.wav
          city/traffic.wav
          music/dramatic.wav
          ...
    """

    def __init__(self):
        self.index: dict[str, Path] = {}   # keyword → caminho
        self._scan()

    def _scan(self):
        """Percorre SOUNDS_DIR e indexa todos os ficheiros de áudio."""
        if not SOUNDS_DIR.exists():
            logger.warning(f"Pasta de sons não encontrada: {SOUNDS_DIR}")
            return

        for ext in ("*.wav", "*.mp3", "*.ogg", "*.flac"):
            for path in SOUNDS_DIR.rglob(ext):
                # Chave: nome sem extensão, minúsculas, sem underscores
                key = path.stem.lower().replace("_", " ").replace("-", " ")
                self.index[key] = path
                # Também indexa pelo nome original (com underscore)
                self.index[path.stem.lower()] = path

        logger.info(f"SoundDB: {len(self.index)} entradas indexadas de {SOUNDS_DIR}")

    def reload(self):
        self.index.clear()
        self._scan()

    def find(self, keyword: str, threshold: float = 0.5) -> Path | None:
        """
        Retorna o Path do som mais próximo de `keyword`.
        Primeiro tenta correspondência exata, depois fuzzy.
        """
        kw = keyword.lower().strip()

        # 1. Correspondência exata
        if kw in self.index:
            return self.index[kw]

        # 2. Contém a keyword
        for key, path in self.index.items():
            if kw in key or key in kw:
                return path

        # 3. Fuzzy match
        best_score, best_path = 0.0, None
        for key, path in self.index.items():
            score = SequenceMatcher(None, kw, key).ratio()
            if score > best_score:
                best_score, best_path = score, path

        if best_score >= threshold:
            return best_path

        # 4. Procura por categoria semântica
        return self._category_fallback(kw)

    def _category_fallback(self, kw: str) -> Path | None:
        """Se não encontra o som exato, tenta um da mesma categoria."""
        for category, keywords in SOUND_CATEGORIES.items():
            for cat_kw in keywords:
                if cat_kw in kw or kw in cat_kw:
                    # Procura qualquer ficheiro nessa categoria
                    cat_dir = SOUNDS_DIR / category
                    if cat_dir.exists():
                        files = list(cat_dir.glob("*.wav")) + list(cat_dir.glob("*.mp3"))
                        if files:
                            return files[0]
        return None

    def list_all(self) -> list[dict]:
        """Lista todos os sons únicos com caminho e categoria."""
        seen = set()
        result = []
        for key, path in self.index.items():
            if path not in seen:
                seen.add(path)
                category = path.parent.name if path.parent != SOUNDS_DIR else "geral"
                result.append({
                    "key": path.stem,
                    "category": category,
                    "path": str(path),
                })
        return sorted(result, key=lambda x: (x["category"], x["key"]))

    @property
    def total(self) -> int:
        return len(set(self.index.values()))


# Instância global (singleton) — importada por outros módulos
_db: SoundDB | None = None

def get_db() -> SoundDB:
    global _db
    if _db is None:
        _db = SoundDB()
    return _db
