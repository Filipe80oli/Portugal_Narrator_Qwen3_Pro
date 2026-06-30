# core/name_mapper.py
from typing import Dict, List

class NameMapper:
    def __init__(self, aliases: Dict[str, List[str]] = None):
        self.aliases = aliases or {}
        self._build_reverse_map()

    def _build_reverse_map(self):
        self.reverse_map = {}
        for cid, terms in self.aliases.items():
            for term in terms:
                self.reverse_map[term.lower()] = cid

    def map_generic(self, term: str, context: str = "") -> str:
        term_lower = term.lower().strip()
        # 1. Correspondência exata
        if term_lower in self.reverse_map:
            return self.reverse_map[term_lower]
        # 2. Tentar encontrar no contexto
        for pattern, cid in self.reverse_map.items():
            if pattern in context.lower():
                return cid
        return "narrator"