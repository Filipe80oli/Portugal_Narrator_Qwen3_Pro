# core/post_processor.py
import logging
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

def clean_character_descriptions(characters: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Melhora descrições para serem úteis ao TTS. Substitui descrições não vocais.

    BUG CORRIGIDO: a versão anterior tinha branches hardcoded para nomes
    específicos de livros de teste ("clyde", "asa", "elvira", "hester",
    "vagabundo") com idades e sotaques inventados. Isto não generaliza para
    nenhum outro livro. Agora usa apenas o género já inferido a partir do
    cid/descrição existente, sem assumir qual personagem é.
    """
    cleaned = {}
    for cid, cdata in characters.items():
        if cid == "narrator":
            cleaned[cid] = cdata
            continue
        desc = cdata.get("description", "")
        voice_keywords = ["voz", "tom", "sotaque", "grave", "aguda", "suave", "autoritário", "rouca", "serena", "hesitante"]
        if not any(keyword in desc.lower() for keyword in voice_keywords):
            # Tentar inferir género a partir de pistas no próprio cid/nome
            cid_lower = cid.lower()
            feminine_hints = ["a_", "mae", "mãe", "irma", "irmã", "tia", "avo_", "menina", "rapariga", "senhora", "mulher"]
            masculine_hints = ["o_", "pai", "irmao", "irmão", "tio", "avo", "menino", "rapaz", "senhor", "homem"]
            if any(h in cid_lower for h in feminine_hints):
                gender = "feminina"
            elif any(h in cid_lower for h in masculine_hints):
                gender = "masculina"
            else:
                gender = "neutra"
            new_desc = f"Voz {gender}, português de Portugal, tom neutro."
            cdata["description"] = new_desc
        cleaned[cid] = cdata
    return cleaned

def merge_duplicate_characters(characters: Dict[str, Dict[str, Any]], segments: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Funde personagens duplicadas.

    BUG CORRIGIDO: a versão anterior tinha um `merge_map` com nomes
    hardcoded de livros de teste específicos (ex: "elvira_griffiths",
    "asa_griffiths", "hester" — de "Uma Tragédia Americana" e de outro
    livro diferente). Para qualquer livro diferente destes, isto:
    (a) não fazia nada de útil, ou
    (b) pior — se o livro tivesse por coincidência uma personagem chamada
        literalmente "rapariga", "pai" ou "hester", fundia-a incorretamente
        com um alvo de outro livro que nem existe nesta análise.

    A resolução real de termos genéricos ("pai", "mãe", "a mulher", etc.)
    já é feita corretamente por `resolve_generic_ids` + `NameMapper`, que
    usa os aliases extraídos do próprio livro pelo Ollama — não nomes fixos.

    Esta função fica agora limitada à sua responsabilidade segura e genérica:
    fundir apenas IDs que são variantes triviais do mesmo (espaços, maiúsculas/
    minúsculas), nunca mapear um termo genérico para um nome específico.
    """
    # Construir mapa de IDs normalizados (lowercase, sem espaços extra) → ID canónico
    normalized_to_canonical: Dict[str, str] = {}
    rename_map: Dict[str, str] = {}

    for cid in list(characters.keys()):
        norm = cid.strip().lower().replace(" ", "_")
        if norm in normalized_to_canonical:
            # Já existe um ID canónico para esta variante — fundir nele
            canonical = normalized_to_canonical[norm]
            rename_map[cid] = canonical
        else:
            normalized_to_canonical[norm] = cid

    for old_id, canonical_id in rename_map.items():
        if old_id == canonical_id:
            continue
        old_desc = characters.get(old_id, {}).get("description", "")
        canon_desc = characters.get(canonical_id, {}).get("description", "")
        # Manter a descrição mais informativa (mais longa) entre as duas variantes
        if len(old_desc) > len(canon_desc):
            characters[canonical_id]["description"] = old_desc
        if old_id in characters:
            del characters[old_id]

    if rename_map:
        for seg in segments:
            cid = seg.get("character_id", "")
            if cid in rename_map:
                seg["character_id"] = rename_map[cid]

    return characters, segments

def post_process_characters_and_segments(characters: Dict[str, Dict[str, Any]], segments: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    characters = clean_character_descriptions(characters)
    characters, segments = merge_duplicate_characters(characters, segments)
    return characters, segments

def post_process_analysis_universal(
    characters: Dict[str, Dict[str, Any]],
    segments: List[Dict[str, Any]],
    normalize: bool = True,
    merge_duplicates: bool = True,
    enhance_descriptions: bool = True
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    if "narrator" not in characters:
        characters["narrator"] = {
            "name": "Narrador",
            "type": "narrator",
            "description": "Voz masculina madura, português de Portugal"
        }
    if normalize:
        from core.text_normalizer import clean_and_normalize_segments
        segments = clean_and_normalize_segments(segments)
        from core.ollama_analyzer import map_emotion
        for seg in segments:
            if "emotion" in seg:
                seg["emotion"] = map_emotion(seg["emotion"])
            if "character_id" in seg and seg["character_id"] not in characters:
                seg["character_id"] = "narrator"
    if enhance_descriptions:
        characters = clean_character_descriptions(characters)
    if merge_duplicates:
        characters, segments = merge_duplicate_characters(characters, segments)
    used_ids = {seg.get("character_id") for seg in segments if seg.get("character_id")}
    removed_chars = [cid for cid in characters if cid != "narrator" and cid not in used_ids]
    for cid in removed_chars:
        del characters[cid]
    if removed_chars:
        logger.info(f"🧹 Personagens sem segmentos associados removidas: {removed_chars}")
    return characters, segments

def resolve_alias_conflicts(aliases: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Remove aliases que estão atribuídos a mais de uma personagem.
    Isso força o NameMapper a resolver o termo pelo contexto, não por um mapeamento fixo.
    """
    # Construir mapa inverso: termo → lista de personagens que o usam
    term_to_chars = {}
    for cid, terms in aliases.items():
        for term in terms:
            term_to_chars.setdefault(term, []).append(cid)

    # Identificar termos com conflito (usados por mais de uma personagem)
    conflicting_terms = {term for term, chars in term_to_chars.items() if len(chars) > 1}

    # Remover esses termos de todos os aliases
    cleaned_aliases = {}
    for cid, terms in aliases.items():
        cleaned_terms = [t for t in terms if t not in conflicting_terms]
        if cleaned_terms:
            cleaned_aliases[cid] = cleaned_terms

    return cleaned_aliases