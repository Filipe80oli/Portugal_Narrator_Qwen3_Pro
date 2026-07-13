# core/post_processor.py
import re
import logging
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

# Hífen isolado por espaço (início "- " ou meio " - ") usado como substituto
# do travessão tipográfico "—" em muitos ebooks convertidos (comum em textos
# PT-BR, ex.: "- Vou já - disse ela."). Só considera hífen isolado por espaço
# em ambos os lados (ou início de segmento seguido de espaço) — nunca hífen
# colado a letras, para não estragar palavras compostas ("arco-íris") ou
# intervalos numéricos ("10-15").
_DIALOGUE_HYPHEN_PATTERN = re.compile(r'(^|\s)-(?=\s)')


def normalize_dialogue_dashes(text: str) -> str:
    """
    Normaliza hífens usados como travessão de diálogo para o travessão
    tipográfico "—", para que toda a lógica existente neste módulo (que
    procura literalmente "—" para detetar/dividir fala vs. narração/tag)
    funcione também em livros que usam "-" em vez de "—". Idempotente:
    texto já com "—" e sem hífens isolados não é alterado.
    """
    if not text or "-" not in text:
        return text
    return _DIALOGUE_HYPHEN_PATTERN.sub(lambda m: m.group(1) + "—", text)


def normalize_segment_dashes(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aplica normalize_dialogue_dashes ao texto de todos os segmentos, in place."""
    changed = 0
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        text = seg.get("text", "")
        new_text = normalize_dialogue_dashes(text)
        if new_text != text:
            seg["text"] = new_text
            changed += 1
    if changed:
        logger.info(
            f"🧹 {changed} segmento(s) com hífen de diálogo normalizado para travessão '—'."
        )
    return segments


def fix_narration_misattributed_as_speech(
    characters: Dict[str, Dict[str, Any]],
    segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Deteta e corrige segmentos onde NARRAÇÃO sobre uma personagem foi
    incorretamente atribuída como sendo a FALA dessa personagem.

    Exemplo real encontrado: um segmento com o texto "E então Burton Burleigh
    concluiu que talvez não tivesse sido o tripé..." (frase em 3ª pessoa,
    claramente a descrever o raciocínio de Burton Burleigh, não uma citação
    dele) foi atribuído a character_id="burton_burleigh" como se fosse a
    sua própria fala direta.

    Heurística (conservadora, para evitar falsos positivos):
    Só reclassifica para "narrator" se AMBAS as condições se verificarem:
    1. O nome completo da personagem (≥2 palavras) aparece literalmente
       dentro do próprio texto do segmento.
    2. O segmento NÃO contém nenhum marcador típico de discurso direto em
       PT-PT (aspas, aspas angulares, travessão de diálogo).
    A ausência de qualquer marcador de diálogo, combinada com a presença do
    próprio nome da personagem no texto, é o sinal de que se trata de uma
    frase narrativa sobre a personagem, não dita por ela.
    """
    dialogue_markers = re.compile(r'["“”«»]|—')
    fixed_count = 0

    for seg in segments:
        cid = seg.get("character_id")
        if not cid or cid == "narrator":
            continue
        cdata = characters.get(cid)
        if not isinstance(cdata, dict):
            continue
        name = cdata.get("name", "").strip()
        if not name or len(name.split()) < 2:
            continue  # nome de uma só palavra é demasiado ambíguo para esta verificação

        text = seg.get("text", "")
        if not text:
            continue

        if name.lower() in text.lower() and not dialogue_markers.search(text):
            seg["character_id"] = "narrator"
            fixed_count += 1

    if fixed_count:
        logger.info(
            f"🧹 {fixed_count} segmento(s) de narração indevidamente atribuídos "
            f"a uma personagem foram repostos para 'narrator'."
        )

    return segments


# Verbos dicendi/cogitandi comuns em PT-PT usados para relatar (não citar) fala ou pensamento
_VERBA_DICENDI = (
    r'disse|respondeu|perguntou|pensou|refletiu|sugeriu|admitiu|confessou|'
    r'murmurou|gritou|exclamou|replicou|acrescentou|insistiu|concordou|negou|'
    r'afirmou|declarou|sussurrou|retorquiu|contou|avisou|garantiu|prometeu|'
    r'jurou|reconheceu|explicou|observou|comentou'
)

# Verbo dicendi seguido (a curta distância) de "que"/"se" → discurso INDIRETO
# (relato do narrador), não citação direta. Ex.: "disse que viria", "perguntou se ela sabia".
_INDIRECT_SPEECH_PATTERN = re.compile(
    rf'\b(?:{_VERBA_DICENDI})\b(?:\s+\S+){{0,4}}?\s+\b(que|se)\b',
    re.IGNORECASE
)

# Pronome de 1ª pessoa perto do verbo → provavelmente é a própria personagem-narradora
# a relatar-se a si mesma (capítulo em 1ª pessoa), não deve ser corrigido para "narrator".
_FIRST_PERSON_GUARD = re.compile(
    rf'\b(eu|me|minha|meu|comigo)\b(?:\s+\S+){{0,3}}?\s+\b(?:{_VERBA_DICENDI})\b',
    re.IGNORECASE
)


def fix_indirect_speech_misattributed(
    characters: Dict[str, Dict[str, Any]],
    segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Complementa `fix_narration_misattributed_as_speech`: aquela função só deteta
    o caso em que o NOME COMPLETO (≥2 palavras) da personagem aparece no texto.
    Esta cobre o caso, mais comum, em que a frase usa um pronome ("ela", "ele")
    ou um verbo dicendi ("disse que...", "perguntou se...") em vez do nome —
    ou seja, discurso INDIRETO relatado pelo narrador, que o LLM por vezes
    atribui erradamente à personagem mencionada em vez de a "narrator".

    Heurística (conservadora):
    Reclassifica para "narrator" apenas se TODAS as condições se verificarem:
    1. O segmento está atribuído a uma personagem específica (não "narrator").
    2. O texto contém um verbo dicendi seguido de "que"/"se" a curta distância
       (padrão típico de discurso indireto: "disse que", "perguntou se").
    3. O texto NÃO contém nenhum marcador de discurso direto (aspas, travessão).
    4. O texto NÃO tem um pronome de 1ª pessoa junto ao verbo (o que indicaria
       que é a própria personagem-narradora a relatar-se, em capítulo na 1ª pessoa).
    """
    dialogue_markers = re.compile(r'["“”«»]|—')
    fixed_count = 0

    for seg in segments:
        cid = seg.get("character_id")
        if not cid or cid == "narrator":
            continue
        text = seg.get("text", "")
        if not text:
            continue

        if not _INDIRECT_SPEECH_PATTERN.search(text):
            continue
        if dialogue_markers.search(text):
            continue
        if _FIRST_PERSON_GUARD.search(text):
            continue

        seg["character_id"] = "narrator"
        fixed_count += 1

    if fixed_count:
        logger.info(
            f"🧹 {fixed_count} segmento(s) de discurso indireto indevidamente "
            f"atribuídos a uma personagem foram repostos para 'narrator'."
        )

    return segments


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
    # Normalizar hífens de diálogo ("- fala -") para travessão "—" ANTES de
    # qualquer deteção/split baseado em "—", para que livros que usam este
    # formato (comum em ebooks convertidos) sejam cobertos pelas heurísticas.
    segments = normalize_segment_dashes(segments)
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
    segments = fix_narration_misattributed_as_speech(characters, segments)
    segments = fix_indirect_speech_misattributed(characters, segments)
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

# Marcadores de 1ª pessoa: usados para decidir se um trecho de narração
# pertence à personagem que está a narrar a cena (comum em livros com POV
# alternado por capítulo, ex.: capítulos contados por Lívia e por Paulo),
# em vez do "narrator" genérico do sistema.
_first_person_tag_markers = re.compile(
    r'\b(eu|me|minha|minhas|meu|meus|comigo)\b', re.IGNORECASE
)


def _looks_like_first_person_tag(tag_text: str) -> bool:
    """
    Heurística conservadora para decidir se um trecho de narração/tag
    pertence à personagem (1ª pessoa) em vez do narrador (3ª pessoa).

    1. Se contém pronome/possessivo de 1ª pessoa em qualquer parte do
       texto (ex.: "...segurando minha mão.") → 1ª pessoa.
    2. Senão, olha para o primeiro verbo: em português, a 1ª pessoa do
       presente do indicativo termina tipicamente em "-o" (falo, penso,
       abraço, arqueio, questiono) e o pretérito perfeito em "-ei"/"-i"
       (falei, respondi); a 3ª pessoa termina em "-a"/"-e" (presente:
       implora, elogia, exige) ou "-ou"/"-eu" (pretérito: disse,
       respondeu, murmurou). Esta distinção não é perfeita (há exceções
       e verbos irregulares), mas evita o erro mais grave: reatribuir em
       bloco tags de 1ª pessoa a "narrator" em livros narrados na 1ª pessoa.
    """
    if _first_person_tag_markers.search(tag_text):
        return True
    match = re.match(r"[^\wÀ-ÿ]*([A-Za-zÀ-ÿ]+)", tag_text)
    if not match:
        return False
    word = match.group(1).lower()
    return len(word) > 3 and (word.endswith("o") or word.endswith("ei") or word.endswith("i"))


def fix_first_person_narration_misattributed_to_narrator(
    characters: Dict[str, Dict[str, Any]],
    segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Complementa `_looks_like_first_person_tag` para além dos segmentos
    divididos por `_split_combined_segment`: cobre também segmentos JÁ
    autónomos (não vieram de nenhum split, o próprio LLM devolveu-os assim)
    que o LLM atribuiu a "narrator" seguindo a regra geral "narração =
    narrador", mas que na verdade são a ação/reação da PRÓPRIA personagem
    que está a narrar aquela cena (ex.: "Arqueio uma sobrancelha, confuso.",
    "questiono, com um sorriso enorme."), típico de livros com POV alternado
    por capítulo.

    Heurística (conservadora): se um segmento atribuído a "narrator" tem
    marcadores claros de 1ª pessoa, só reatribui à personagem não-narrador
    quando a personagem não-narrador ANTERIOR e a SEGUINTE na sequência
    concordam (é a mesma). Isto evita amplificar um erro de atribuição já
    existente vindo de trás (ex.: se um segmento anterior já estava mal
    atribuído a uma personagem errada por outra causa, usar só "a anterior"
    como âncora propagaria esse erro para a frente; exigir também que a
    seguinte concorde reduz bastante esse risco). Numa conversa entre duas
    personagens, se a anterior e a seguinte forem falantes DIFERENTES, é
    sinal de que estamos numa troca de diálogo ambígua — mantém-se "narrator"
    em vez de arriscar um palpite.
    """
    fixed_count = 0
    n = len(segments)

    # Pré-computar, para cada posição, a personagem não-narrador mais próxima
    # a seguir (evita repetir a procura O(n) para cada segmento narrator).
    next_non_narrator = [None] * n
    upcoming = None
    for i in range(n - 1, -1, -1):
        next_non_narrator[i] = upcoming
        cid = segments[i].get("character_id")
        if cid and cid != "narrator":
            upcoming = cid

    last_non_narrator_cid = None
    for idx, seg in enumerate(segments):
        cid = seg.get("character_id")
        if cid and cid != "narrator":
            last_non_narrator_cid = cid
            continue
        if cid != "narrator" or last_non_narrator_cid is None:
            continue
        text = seg.get("text", "")
        if not text or not _looks_like_first_person_tag(text):
            continue
        forward_cid = next_non_narrator[idx]
        if forward_cid is not None and forward_cid != last_non_narrator_cid:
            # Anterior e seguinte discordam — ambíguo, não arriscar palpite.
            continue
        seg["character_id"] = last_non_narrator_cid
        fixed_count += 1

    if fixed_count:
        logger.info(
            f"🧹 {fixed_count} segmento(s) de narração em 1ª pessoa indevidamente "
            f"atribuídos a 'narrator' foram repostos para a personagem que narra a cena."
        )
    return segments


def fix_known_errors(segments: List[Dict], characters: Dict) -> List[Dict]:
    """
    Corrige erros estruturais comuns: divide segmentos que combinam
    fala e narração, e junta segmentos cortados a meio de frase.
    Esta versão é universal e não depende de nomes de personagens específicos.
    """
    # Defensivo: normaliza hífens de diálogo também aqui, para o caso de esta
    # função ser chamada diretamente sobre segmentos/cache que não passaram
    # por post_process_analysis_universal (ex.: análises antigas recarregadas).
    segments = normalize_segment_dashes(segments)

    def _split_combined_segment(text: str, cid: str, emotion: str) -> List[Dict[str, str]]:
        """
        Divide um segmento que combina narração e fala (separadas por "—").
        Retorna lista de dicionários com "text" e "character_id".

        BUG CORRIGIDO (1/2): a versão anterior tratava apenas o primeiro pedaço
        (antes do 1º travessão) como narração, e TUDO o resto como fala da
        personagem. Isto está errado para o padrão típico do diálogo em
        PT-PT, que alterna: "— fala — disse ele, saindo. — mais fala."
        Aqui "disse ele, saindo." (entre o 2º e o 3º travessão) é narração/
        tag, não continuação da fala da personagem — mas a versão anterior
        atribuía-o à personagem na mesma. A convenção correta é alternância
        por paridade: cada pedaço PAR (0, 2, 4...) é narração/tag; cada
        pedaço ÍMPAR (1, 3, 5...) é fala.

        BUG CORRIGIDO (2/2): mas "narração/tag" não é sempre "narrator"!
        Em livros narrados na 1ª pessoa (comum em contos/novelas em que o
        capítulo é contado por uma das personagens, ex.: "Abraço-a forte...",
        "sussurro em seu ouvido..."), a tag entre travessões é a própria
        ação/reação da personagem-narradora daquela cena, não do "narrator"
        genérico do sistema — atribuí-la a "narrator" trocaria a voz a meio
        da cena. Por isso cada pedaço PAR só vai para "narrator" se parecer
        claramente descrição em 3ª pessoa; se parecer 1ª pessoa, mantém-se
        com o `cid` original do segmento (ver `_looks_like_first_person_tag`).
        """
        parts = []
        segments = text.split("—")
        for i, part in enumerate(segments):
            part = part.strip()
            if not part:
                continue
            if i % 2 == 0:
                if _looks_like_first_person_tag(part):
                    parts.append({"text": part, "character_id": cid})
                else:
                    parts.append({"text": part, "character_id": "narrator"})
            else:
                parts.append({"text": part, "character_id": cid})
        return parts if parts else [{"text": text, "character_id": cid}]

    def merge_split_segments(seg_list: List[Dict]) -> List[Dict]:
            """
            Junta segmentos consecutivos cortados a meio de uma frase.
            Critério: o anterior termina com palavra incompleta (sem pontuação final)
            e o seguinte começa com minúscula.

            BUG CORRIGIDO: faltava verificar se os dois segmentos pertencem ao
            MESMO falante. Sem essa verificação, isto desfazia o split correto
            de "— fala — tag." (ex.: "Eu disse que viria mais tarde" [maria] +
            "respondeu ela." [narrator] cumpre o critério de "frase cortada" —
            sem pontuação final + minúscula a seguir — e era remendado de volta
            num único segmento, perdendo a separação fala/narração). Esta função
            destina-se apenas a juntar cortes de bloco dentro da MESMA voz.
            """
            merged = []
            i = 0
            while i < len(seg_list):
                if i + 1 < len(seg_list):
                    cur = seg_list[i].get("text", "").strip()
                    nxt = seg_list[i+1].get("text", "").strip()
                    same_speaker = seg_list[i].get("character_id") == seg_list[i+1].get("character_id")
                    if (same_speaker and cur and nxt
                            and not re.search(r'[.!?…]\s*$', cur) and nxt[0].islower()):
                        seg_list[i]["text"] = cur + " " + nxt
                        del seg_list[i+1]
                        continue
                merged.append(seg_list[i])
                i += 1
            return merged

        # ── Processar cada segmento ──────────────────────────────────────────────
    corrected = []

    for seg in segments:
            if not isinstance(seg, dict):
                corrected.append(seg)
                continue

            text = seg.get("text", "")
            cid = seg.get("character_id", "narrator")
            emotion = seg.get("emotion", "neutral")
            pace = seg.get("pace", 1.0)
            pause_ms = seg.get("pause_ms", 0)

            # ── Dividir segmentos que combinam fala e narração ──────────────────
            # BUG CORRIGIDO: a condição anterior só dividia quando o texto NÃO
            # começava por travessão, partindo do princípio de que "começa com
            # travessão" = "é só fala, sem nada a separar". Isso está errado
            # para o padrão mais comum em PT-PT: "— fala — tag/narração.",
            # que começa por travessão MAS tem uma tag narrativa no fim (ex.:
            # "— Eu já vou — respondeu ela.") que ficava indevidamente colada
            # à fala da personagem. Agora divide-se sempre que há travessão;
            # _split_combined_segment lida corretamente com o pedaço vazio
            # inicial quando o texto começa por "—".
            if "—" in text:
                split_parts = _split_combined_segment(text, cid, emotion)
                for part in split_parts:
                    corrected.append({
                        "text": part["text"],
                        "character_id": part["character_id"],
                        "emotion": emotion,
                        "pace": pace,
                        "pause_ms": pause_ms
                    })
            else:
                # Manter o segmento, mas com character_id possivelmente genérico
                seg["character_id"] = cid
                corrected.append(seg)

        # ── Juntar segmentos cortados ──────────────────────────────────────────
    corrected = merge_split_segments(corrected)
    corrected = fix_first_person_narration_misattributed_to_narrator(characters, corrected)

    return corrected