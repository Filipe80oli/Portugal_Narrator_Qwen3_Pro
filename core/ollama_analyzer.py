# core/ollama_analyzer.py
"""
Análise do livro via Ollama – Versão com separação em duas fases:
Fase 1 – Extração de personagens (varredura rápida)
Fase 2 – Segmentação com elenco conhecido
"""

import re
import json
import logging
import asyncio
import requests
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════
TIMEOUT_SMALL = 120
TIMEOUT_MEDIUM = 300
TIMEOUT_LARGE = 600
MAX_BLOCK_SIZE = 1500  # tamanho padrão para a Fase 2 (segmentação)
MAX_BLOCK_SIZE_CHAR_EXTRACTION = 6000  # tamanho maior para a Fase 1 (personagens)

# Padrão para títulos/cabeçalhos (forçar emotion: neutral)
TITLE_PATTERN = re.compile(
    r'^(sem direitos de autor|prefácio|introdução|capítulo|parte|livro|índice|nota|epílogo|apêndice|título|sumário)',
    re.IGNORECASE
)

# ═══════════════════════════════════════════════════════════════════════════
# MAPEAMENTO DE EMOÇÕES (PT/ESTRANHO → EN)
# ═══════════════════════════════════════════════════════════════════════════
EMOTION_MAP = {
    "neutral": "neutral", "neutro": "neutral", "racional": "neutral",
    "insincero": "neutral", "fantasioso": "neutral", "explicativo": "neutral",
    "informativo": "neutral", "informative": "neutral", "reflexivo": "neutral",
    "reflective": "neutral", "curioso": "neutral", "curious": "neutral",
    "inquisitivo": "neutral", "inquisitive": "neutral", "questioning": "neutral",
    "questionador": "neutral", "reservado": "neutral", "cético": "neutral",
    "skeptical": "neutral", "sério": "neutral", "serious": "neutral",
    "persuasivo": "neutral", "persuasive": "neutral", "instrutivo": "neutral",
    "instructive": "neutral", "planejador": "neutral", "detalhista": "neutral",
    "interessado": "neutral", "interested": "neutral", "amigável": "neutral",
    "informal": "neutral", "superioridade": "neutral", "firme": "neutral",
    "honesto": "neutral", "honest": "neutral", "interrogativo": "neutral",
    "perspicaz": "neutral", "prático": "neutral", "practical": "neutral",
    "compassivo": "neutral", "solidária": "neutral", "solidario": "neutral",
    "calm": "calm", "calmo": "calm", "calma": "calm", "reassurante": "calm",
    "reassuring": "calm", "confiante": "calm", "confident": "calm",
    "submissive": "calm", "submisso": "calm", "polite": "calm",
    "respeitoso": "calm", "devoto": "calm", "devout": "calm",
    "afetuoso": "calm", "affectionate": "calm", "romântico": "calm",
    "romantic": "calm", "aliviado": "calm", "relieved": "calm",
    "consolo": "calm", "comforting": "calm", "encourajador": "calm",
    "encouraging": "calm", "convencido": "calm", "cooperativo": "calm",
    "cooperative": "calm", "diretivo": "calm", "directive": "calm",
    "calm_and_directive": "calm", "sereno": "calm", "confortante": "calm",
    "segura": "calm", "seguro": "calm", "aceitação": "calm",
    "tense": "tense", "tenso": "tense", "tensa": "tense", "tensão": "tense",
    "preocupado": "tense", "preocupada": "tense", "ansioso": "tense",
    "ansiosa": "tense", "anxious": "tense", "ansiedade": "tense",
    "anxiety": "tense", "nervoso": "tense", "nervosa": "tense",
    "nervous": "tense", "dúvida": "tense", "doubt": "tense",
    "indeciso": "tense", "indecisive": "tense", "hesitante": "tense",
    "hesitant": "tense", "relutante": "tense", "reluctant": "tense",
    "constrangido": "tense", "defensivo": "tense", "defensive": "tense",
    "defensiva": "tense", "determinado": "tense", "determinada": "tense",
    "determined": "tense", "decidido": "tense", "urgente": "tense",
    "urgency": "tense", "urgent": "tense", "urgência": "tense",
    "cautionary": "tense", "cauteloso": "tense", "cautious": "tense",
    "carrancudo": "tense", "desconcertado": "tense", "disconcerted": "tense",
    "perplexo": "tense", "perplexed": "tense", "confuso": "tense",
    "confused": "tense", "confusão": "tense", "frustrado": "tense",
    "frustrated": "tense", "desafiador": "tense", "defiant": "tense",
    "agressivo": "tense", "aggressive": "tense", "feroz": "tense",
    "ferocious": "tense", "indignado": "tense", "indignant": "tense",
    "acusador": "tense", "accusatory": "tense", "mentiroso": "tense",
    "ativação": "tense", "sonhador": "tense", "apaixonado": "tense",
    "sedutora e calculista": "tense", "doubtful": "tense",
    "incerto": "tense", "uncertain": "tense", "incerteza": "tense",
    "insegura": "tense", "inseguro": "tense", "zangada": "tense",
    "ciumento": "tense", "ciumenta": "tense", "expectante": "tense",
    "insistente": "tense", "emphatic": "tense", "assertive": "tense",
    "joyful": "joyful", "alegre": "joyful", "feliz": "joyful", "happy": "joyful",
    "enthusiastic": "joyful", "excited": "joyful", "excitado": "joyful",
    "amused": "joyful", "divertido": "joyful", "playful": "joyful",
    "brincalhão": "joyful", "brincalhona": "joyful", "esperançoso": "joyful",
    "hopeful": "joyful", "esperançosa": "joyful", "inspirado": "joyful",
    "inspired": "joyful", "impressionado": "joyful", "impressed": "joyful",
    "surpreso": "joyful", "surprised": "joyful", "bajuladora": "joyful",
    "flattering": "joyful", "loving": "joyful", "amoroso": "joyful",
    "nostálgico": "joyful", "nostalgic": "joyful", "entusiasmado": "joyful",
    "entusiasmada": "joyful", "jovial": "joyful", "teasing": "joyful",
    "charmed": "joyful", "flattered": "joyful", "passionate": "joyful",
    "ironia": "joyful", "ironic": "joyful", "sarcástico": "joyful",
    "sad": "sad", "triste": "sad", "tristeza": "sad", "sadness": "sad",
    "desesperado": "sad", "desesperada": "sad", "desperate": "sad",
    "desespero": "sad", "angustiado": "sad", "anguished": "sad",
    "implorante": "sad", "pleading": "sad", "envergonhado": "sad",
    "ashamed": "sad", "culpado": "sad", "guilty": "sad", "culpa": "sad",
    "arrependido": "sad", "regretful": "sad", "arrependimento": "sad",
    "desapontado": "sad", "disappointed": "sad", "desapontamento": "sad",
    "cansado": "sad", "tired": "sad", "desolado": "sad", "desolate": "sad",
    "resignado": "sad", "resigned": "sad", "medo": "sad", "solidão": "sad",
    "lonely": "sad", "devastado": "sad", "aceitação": "sad",
    "angry": "angry", "zangado": "angry", "irritado": "angry",
    "irritada": "angry", "irritated": "angry", "annoyed": "angry",
    "fearful": "fearful", "assustado": "fearful", "medo": "fearful",
    "terrified": "fearful", "alarmado": "fearful", "alarmed": "fearful",
    "whisper": "whisper", "sussurro": "whisper", "sussurrando": "whisper"
}

def map_emotion(raw_emotion: str) -> str:
    """Mapeia emoções, lidando com compostas como 'incerteza e ansiedade'."""
    if not raw_emotion:
        return "neutral"
    raw = raw_emotion.lower().strip()
    if raw in EMOTION_MAP:
        return EMOTION_MAP[raw]
    for word in re.split(r'[\s,]+', raw):
        if word in EMOTION_MAP:
            return EMOTION_MAP[word]
    return "neutral"


def repair_truncated_json(raw: str) -> str:
    """Repara JSON truncado fechando estruturas abertas na ordem correta."""
    if not raw.strip():
        return raw
    raw = raw.strip()

    # PASSO 1: Cortar último item incompleto (antes da vírgula)
    in_string = False
    escape_next = False
    last_comma_pos = -1
    for i, ch in enumerate(raw):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == ',':
            last_comma_pos = i

    if last_comma_pos > 0:
        candidate = raw[:last_comma_pos]
        # tentar fechar
        stack = []
        in_string2 = False
        esc2 = False
        for ch in candidate:
            if esc2:
                esc2 = False
                continue
            if ch == '\\' and in_string2:
                esc2 = True
                continue
            if ch == '"':
                in_string2 = not in_string2
                continue
            if in_string2:
                continue
            if ch in ('{', '['):
                stack.append(ch)
            elif ch in ('}', ']'):
                if stack:
                    stack.pop()
        closing = ''
        for opener in reversed(stack):
            closing += '}' if opener == '{' else ']'
        repaired = candidate + closing
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            pass

    # PASSO 2: Fechar com stack original (sem cortar)
    stack = []
    in_string = False
    esc = False
    for ch in raw:
        if esc:
            esc = False
            continue
        if ch == '\\' and in_string:
            esc = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ('{', '['):
            stack.append(ch)
        elif ch in ('}', ']'):
            if stack:
                stack.pop()
    closing = ''
    for opener in reversed(stack):
        closing += '}' if opener == '{' else ']'
    repaired = raw + closing
    try:
        json.loads(repaired)
        return repaired
    except json.JSONDecodeError:
        pass

    # PASSO 3: Fallback – varrer de trás para a frente
    for i in range(len(raw) - 1, -1, -1):
        if raw[i] in ('}', ']'):
            try:
                json.loads(raw[:i + 1])
                return raw[:i + 1]
            except json.JSONDecodeError:
                continue
    return raw


# ═══════════════════════════════════════════════════════════════════════════
# 1. GESTÃO DE MODELOS OLLAMA
# ═══════════════════════════════════════════════════════════════════════════
def get_ollama_models(base_url: str) -> List[str]:
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception as e:
        logger.warning(f"Ollama indisponível: {e}")
        return ["gemma3:27b"]

def warmup_ollama(ollama_url: str, model_name: str, timeout: int = 120):
    try:
        requests.post(ollama_url, json={
            "model": model_name,
            "prompt": "ok",
            "keep_alive": -1,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 1}
        }, timeout=timeout)
    except Exception as e:
        logger.warning(f"Warmup falhou (não crítico): {e}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. DIVISÃO INTELIGENTE DE TEXTO
# ═══════════════════════════════════════════════════════════════════════════
def split_into_blocks(text: str, max_chars: int = MAX_BLOCK_SIZE) -> List[str]:
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    blocks = []
    current_block = ""
    for para in paragraphs:
        if len(para) > max_chars:
            if current_block:
                blocks.append(current_block)
                current_block = ""
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sentence in sentences:
                if len(current_block) + len(sentence) > max_chars and current_block:
                    blocks.append(current_block)
                    current_block = sentence
                else:
                    current_block += (" " + sentence if current_block else sentence)
        else:
            if len(current_block) + len(para) + 2 > max_chars and current_block:
                blocks.append(current_block)
                current_block = para
            else:
                current_block += ("\n\n" + para if current_block else para)
    if current_block:
        blocks.append(current_block)
    return blocks


# ═══════════════════════════════════════════════════════════════════════════
# 3. SANITIZAÇÃO DE SEGMENTOS
# ═══════════════════════════════════════════════════════════════════════════
def sanitize_segments(raw_segments: List[Any]) -> List[Dict[str, Any]]:
    result = []
    _json_leak_pattern = re.compile(
        r'"(?:character_id|pause_ms|emotion|pace|text|segments|characters)"\s*:'
    )

    def _is_junk(text: str) -> bool:
        if re.match(r'^[\s\.\,\;\:\!\?\-{}\[\]"\'()]+$', text):
            return True
        if _json_leak_pattern.search(text):
            return True
        symbol_chars = sum(1 for c in text if c in '{}[]":')
        if len(text) > 0 and symbol_chars / len(text) > 0.3:
            return True
        if re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u0600-\u06ff]', text):
            return True
        return False

    for item in raw_segments:
        if isinstance(item, dict):
            item = {k.strip(): v for k, v in item.items()}
            text = str(item.get("text", "")).strip()
            if not text or len(text) < 3:
                continue
            if _is_junk(text):
                continue
            character_id = item.get("character_id", "narrator")
            if isinstance(character_id, list):
                character_id = character_id[0] if character_id else "narrator"
            character_id = str(character_id).strip().lower().replace(" ", "_") or "narrator"
            emotion = item.get("emotion", "neutral")
            if isinstance(emotion, list):
                emotion = emotion[0] if emotion else "neutral"
            emotion = str(emotion).strip().lower() or "neutral"
            emotion = map_emotion(emotion)
            if TITLE_PATTERN.match(text):
                emotion = "neutral"
            pace = item.get("pace", 1.0)
            if isinstance(pace, list):
                pace = pace[0] if pace else 1.0
            try:
                pace = float(pace)
                pace = max(0.5, min(2.0, pace))
            except (ValueError, TypeError):
                pace = 1.0
            text_len = len(text)
            if text_len < 30:
                pause_ms = 150
            elif text_len < 80:
                pause_ms = 250
            elif text_len < 200:
                pause_ms = 350
            else:
                pause_ms = 500
            if TITLE_PATTERN.match(text):
                pause_ms = 800
            if re.match(r'^(cap[ií]tulo|livro|parte)\s+', text, re.IGNORECASE):
                pause_ms = 1200
            result.append({
                "text": text,
                "character_id": character_id,
                "emotion": emotion,
                "pace": pace,
                "pause_ms": pause_ms,
            })
        elif isinstance(item, str):
            text = item.strip()
            if text and len(text) >= 3 and not _is_junk(text):
                emotion = "neutral"
                pause_ms = 300
                if TITLE_PATTERN.match(text):
                    pause_ms = 800
                if re.match(r'^(cap[ií]tulo|livro|parte)\s+', text, re.IGNORECASE):
                    pause_ms = 1200
                result.append({
                    "text": text, "character_id": "narrator",
                    "emotion": emotion, "pace": 1.0, "pause_ms": pause_ms
                })
        elif isinstance(item, list):
            result.extend(sanitize_segments(item))
    return result

async def extract_characters_from_text(
    ollama_url: str,
    model_name: str,
    full_text: str,
    max_block_size: int = MAX_BLOCK_SIZE_CHAR_EXTRACTION,
    user_settings: Optional[Dict[str, Any]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Percorre o texto em blocos e extrai todas as personagens.
    Versão com maior robustez a respostas mal formatadas.
    """
    blocks = split_into_blocks(full_text, max_chars=max_block_size)
    all_chars = {}
    total = len(blocks)
    
    # Timeout mais longo para extração (usar large)
    timeout = user_settings.get("ollama_timeout_large", 600) if user_settings else 600

    for idx, block in enumerate(blocks):
        logger.info(f"🔎 Extraindo personagens – bloco {idx+1}/{total} ({len(block)} chars)")
        
        # Prompt simplificado para reduzir a carga do modelo
        prompt = f"""Analisa o seguinte excerto de um livro em português de Portugal.

Tarefa: Identifica TODAS as personagens que aparecem neste excerto (incluindo as que apenas são mencionadas).
Para cada personagem, fornece:
- Nome próprio (como aparece no texto)
- Uma breve descrição da VOZ (género, idade aproximada, tom, sotaque – sempre português de Portugal)

Regras:
- Usa IDs em minúsculas com underscores (ex: "joao_silva").
- Inclui personagens sem nome próprio, usando um ID descritivo (ex: "jovem_empregado").
- A descrição deve ser sobre a voz, nunca sobre aspeto físico.
- Se não houver personagens, devolve {{"personagens": {{}}}}.

Responde APENAS com JSON no formato:
{{"personagens": {{"id1": {{"name": "Nome", "description": "Voz ..."}}, "id2": {{...}}}}}}

Excerto:
""" + block

        # Tentativas com backoff
        for attempt in range(3):  # até 3 tentativas
            try:
                def _make_request():
                    return requests.post(ollama_url, json={
                        "model": model_name,
                        "prompt": prompt,
                        "stream": False,
                        "keep_alive": -1,
                        "options": {
                            "temperature": 0.0,
                            "top_k": 10,
                            "repeat_penalty": 1.1,
                            "num_ctx": 8192 + (attempt * 4096),  # aumenta a cada tentativa
                            "num_predict": 2048 + (attempt * 1024)
                        }
                    }, timeout=timeout + (attempt * 60))  # aumenta timeout

                r = await asyncio.to_thread(_make_request)
                r.raise_for_status()
                raw = r.json().get('response', '').strip()
                
                # Limpeza agressiva
                raw = raw.replace('```json', '').replace('```', '').strip()
                # Remover tudo antes do primeiro '{' e depois do último '}'
                first_brace = raw.find('{')
                last_brace = raw.rfind('}')
                if first_brace != -1 and last_brace != -1:
                    raw = raw[first_brace:last_brace+1]
                else:
                    # Se não encontrar chavetas, tentar extrair com regex
                    match = re.search(r'(\{.*\})', raw, re.DOTALL)
                    if match:
                        raw = match.group(1)
                    else:
                        logger.warning(f"Bloco {idx+1}: nenhum JSON detetado. Resposta: {raw[:200]}...")
                        break  # passa ao próximo bloco

                # Verificar se o JSON está truncado
                if raw.count('{') > raw.count('}') or raw.count('[') > raw.count(']'):
                    raw = repair_truncated_json(raw)
                    logger.debug(f"Bloco {idx+1}: JSON reparado.")

                data = json.loads(raw)
                personagens = data.get("personagens", {})
                if not personagens:
                    logger.info(f"Bloco {idx+1}: nenhuma personagem encontrada.")
                    break  # sucesso, mas vazio

                for cid, cdata in personagens.items():
                    if not isinstance(cdata, dict):
                        continue
                    cid_clean = str(cid).strip().lower().replace(" ", "_")
                    if not cid_clean:
                        continue
                    if "name" not in cdata:
                        cdata["name"] = cid_clean
                    if "description" not in cdata or not cdata["description"].strip():
                        cdata["description"] = f"Voz de {cdata['name']}, português de Portugal, tom neutro."
                    cdata["type"] = "character"
                    # Fundir com o dicionário global (manter descrição mais longa)
                    if cid_clean in all_chars:
                        if len(cdata.get("description", "")) > len(all_chars[cid_clean].get("description", "")):
                            all_chars[cid_clean]["description"] = cdata["description"]
                    else:
                        all_chars[cid_clean] = cdata
                break  # sucesso, sair do loop de tentativas

            except json.JSONDecodeError as e:
                logger.warning(f"Bloco {idx+1}, tentativa {attempt+1}: JSON inválido - {e}")
                if attempt == 2:
                    # Fallback: tentar extrair nomes próprios via regex
                    names = re.findall(r'\b([A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][a-záâãàéêíóôõúç]+)\b', block)
                    for name in set(names):
                        if len(name) < 3 or name.lower() in ('um', 'uma', 'o', 'a', 'os', 'as'):
                            continue
                        cid = name.lower()
                        if cid not in all_chars:
                            all_chars[cid] = {
                                "name": name,
                                "type": "character",
                                "description": f"Voz de {name}, português de Portugal, tom neutro."
                            }
                    logger.info(f"Bloco {idx+1}: fallback – extraídos {len(set(names))} nomes por regex.")
                continue
            except requests.exceptions.Timeout:
                logger.warning(f"Bloco {idx+1}, tentativa {attempt+1}: timeout")
                if attempt == 2:
                    logger.error(f"Bloco {idx+1}: falhou após 3 tentativas (timeout).")
                continue
            except Exception as e:
                logger.warning(f"Bloco {idx+1}, tentativa {attempt+1}: erro - {e}")
                if attempt == 2:
                    logger.error(f"Bloco {idx+1}: falhou após 3 tentativas.")
                continue

    # Garantir que existe narrador
    if "narrator" not in all_chars:
        all_chars["narrator"] = {
            "name": "Narrador",
            "type": "narrator",
            "description": "Voz masculina madura, português de Portugal, tom neutro e sóbrio."
        }

    logger.info(f"✅ Fase 1 concluída: {len(all_chars)} personagens encontradas.")
    return all_chars


# ═══════════════════════════════════════════════════════════════════════════
# 5. FASE 2 – ANÁLISE DE BLOCO COM ELENCO CONHECIDO
# ═══════════════════════════════════════════════════════════════════════════
async def analyze_block(
    ollama_url: str,
    model_name: str,
    text: str,
    context: str,
    known_characters: Optional[Dict[str, Dict[str, Any]]] = None,
    aliases: Optional[Dict[str, List[str]]] = None,
    max_retries: int = 2,
    user_settings: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Segmenta o bloco usando a lista de personagens conhecidas.
    Retorna dicionário com "segments" e, opcionalmente, "characters" (novas personagens).
    """
    if known_characters is None:
        known_characters = {}

    # Construir lista de personagens conhecidas
    char_list_lines = []
    for cid, cdata in known_characters.items():
        if not isinstance(cdata, dict):
            continue
        name = cdata.get("name", cid)
        char_list_lines.append(f"- {cid}: {name}")
    char_list = "\n".join(char_list_lines) if char_list_lines else "(nenhuma personagem conhecida)"

    # Construir alias_text se fornecido
    alias_text = ""
    if aliases:
        alias_lines = []
        for cid, terms in aliases.items():
            if cid in known_characters:
                alias_lines.append(f"- {cid}: {', '.join(terms)}")
        if alias_lines:
            alias_text = "\nALIASES CONHECIDOS (usa estes para mapear termos genéricos):\n" + "\n".join(alias_lines) + "\n"

    context_block = f"CONTEXTO:\n{context[:500]}...\n\n" if context else ""

    # Definir timouts e parâmetros dinâmicos
    text_length = len(text)
    small = user_settings.get("ollama_timeout_small", TIMEOUT_SMALL) if user_settings else TIMEOUT_SMALL
    medium = user_settings.get("ollama_timeout_medium", TIMEOUT_MEDIUM) if user_settings else TIMEOUT_MEDIUM
    large = user_settings.get("ollama_timeout_large", TIMEOUT_LARGE) if user_settings else TIMEOUT_LARGE

    estimated_prompt_tokens = 1500 + (text_length // 4)
    estimated_response_tokens = max(4096, text_length * 2)
    needed_ctx = estimated_prompt_tokens + estimated_response_tokens
    base_num_ctx = max(8192, min(needed_ctx, 32768))
    for p in (8192, 16384, 32768):
        if base_num_ctx <= p:
            base_num_ctx = p
            break
    base_num_predict = min(estimated_response_tokens, base_num_ctx - estimated_prompt_tokens)
    base_num_predict = max(4096, base_num_predict)

    user_num_predict = user_settings.get("ollama_num_predict", 0) if user_settings else 0
    user_num_ctx = user_settings.get("ollama_num_ctx", 0) if user_settings else 0
    if user_num_predict > 0:
        base_num_predict = user_num_predict
    if user_num_ctx > 0:
        base_num_ctx = user_num_ctx

    num_predict = base_num_predict
    num_ctx = base_num_ctx

    if num_ctx <= 8192:
        timeout = small
    elif num_ctx <= 16384:
        timeout = medium
    else:
        timeout = large

    prompt = f"""{context_block}/no_think
Analisa este trecho em PT-PT. Segmenta o texto em unidades de fala ou narração.

PERSONAGENS CONHECIDAS (usa APENAS estes IDs sempre que possível):
{char_list}

{alias_text}

Tarefa: Para cada segmento, indica:
- O ID da personagem que fala (deve ser um dos IDs da lista acima)
- A emoção (neutral, calm, tense, joyful, sad, angry, fearful, whisper)
- Ritmo (pace, entre 0.5 e 2.0)
- Pausa (pause_ms, em milissegundos)

REGRAS:
- O narrador é identificado como "narrator".
- Se uma fala pertencer a uma personagem que NÃO está na lista, cria um novo ID descritivo (ex: "jovem_empregado", "transeunte") e adiciona-o à lista de personagens no campo "characters" da resposta.
- As emoções devem ser inferidas a partir de pistas contextuais (verbos, pontuação, palavras de intensidade).
- **Nunca** atribuas uma fala ao narrador se houver aspas ou travessão a indicar que alguém está a falar.
- Títulos e cabeçalhos devem ser marcados como "narrator" com "neutral".
- Para descrições de voz de novas personagens, segue o padrão: "Voz [género], [idade], [tom], português de Portugal."

EXEMPLOS:
- Texto: — Vamos cantar o hino, disse o pai.
  Saída: {{"character_id": "pai", "emotion": "calm", "pace": 1.0, "pause_ms": 300}}

- Texto: "Vejo estas pessoas por aqui quase todas as noites", disse um jovem empregado.
  Saída: {{"character_id": "jovem_empregado", "emotion": "neutral", "pace": 1.0, "pause_ms": 200}}

- Texto: Capítulo 1 — O Início
  Saída: {{"character_id": "narrator", "emotion": "neutral", "pace": 1.0, "pause_ms": 800}}

Responde APENAS com JSON válido no formato:
{{"segments": [{{"text": "...", "character_id": "...", "emotion": "...", "pace": 1.0, "pause_ms": 0}}], "characters": {{"novo_id": {{"name": "Nome", "description": "Voz ..."}}}}}}

TEXTO A ANALISAR:
""" + text

    # Loop de tentativas
    for attempt in range(max_retries + 1):
        try:
            def _make_request():
                return requests.post(ollama_url, json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": -1,
                    "options": {
                        "temperature": 0.1,
                        "top_k": 20,
                        "repeat_penalty": 1.1,
                        "num_predict": num_predict,
                        "num_ctx": num_ctx,
                        "mirostat": 0
                    }
                }, timeout=timeout)

            r = await asyncio.to_thread(_make_request)
            r.raise_for_status()
            raw = r.json().get('response', '').strip()
            raw = raw.replace('```json', '').replace('```', '').strip()

            # Extrair JSON
            match = re.search(r'(\{.*\})', raw, re.DOTALL)
            if match:
                raw = match.group(1)
            else:
                raw = re.sub(r'[^\{\}\[\]":,0-9a-zA-Z_\-\. ]', '', raw)

            # Verificar truncamento
            open_braces = raw.count('{')
            close_braces = raw.count('}')
            open_brackets = raw.count('[')
            close_brackets = raw.count(']')
            if open_braces > close_braces or open_brackets > close_brackets:
                raw = repair_truncated_json(raw)

            result = json.loads(raw)

            # Processar personagens existentes na resposta
            if "characters" in result:
                valid_chars = {}
                for cid, cdata in result["characters"].items():
                    if isinstance(cdata, dict):
                        cid_str = str(cid).strip().lower().replace(" ", "_")
                        if "name" not in cdata:
                            cdata["name"] = cid_str
                        if "description" not in cdata or not cdata["description"].strip():
                            cdata["description"] = f"Voz de {cdata['name']}, português de Portugal, tom neutro."
                        cdata["type"] = "character"
                        valid_chars[cid_str] = cdata
                result["characters"] = valid_chars

            # Processar segmentos
            if "segments" in result:
                for seg in result["segments"]:
                    if "character_id" not in seg or not seg["character_id"]:
                        seg["character_id"] = "narrator"
                    seg["character_id"] = str(seg["character_id"]).strip().lower().replace(" ", "_")
                    if "emotion" in seg:
                        seg["emotion"] = map_emotion(seg["emotion"])
                    else:
                        seg["emotion"] = "neutral"
                    if "pace" not in seg:
                        seg["pace"] = 1.0
                    if "pause_ms" not in seg:
                        seg["pause_ms"] = 0

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"JSON inválido (tentativa {attempt+1}/{max_retries+1}): {e}")
            if attempt < max_retries:
                num_predict = min(num_predict * 2, 131072)
                num_ctx = min(num_ctx * 2, 65536)
                logger.warning(f"A aumentar num_predict para {num_predict} e num_ctx para {num_ctx}")
                continue
            return {"segments": [], "characters": {}}

        except requests.exceptions.Timeout:
            if attempt < max_retries:
                logger.warning(f"Timeout, tentativa {attempt+1}/{max_retries+1}")
                timeout = min(timeout * 1.5, 600)
                num_predict = min(num_predict * 2, 32768)
                num_ctx = min(num_ctx * 2, 16384)
                continue
            return {"segments": [], "characters": {}}

        except Exception as e:
            logger.warning(f"Erro Ollama: {e}")
            if attempt < max_retries:
                continue
            return {"segments": [], "characters": {}}

    return {"segments": [], "characters": {}}


# ═══════════════════════════════════════════════════════════════════════════
# 6. EXTRAÇÃO DE ALIASES
# ═══════════════════════════════════════════════════════════════════════════
async def extract_aliases(
    ollama_url: str,
    model_name: str,
    full_text: str,
    characters: Dict[str, dict],
    max_retries: int = 2,
    user_settings: Optional[Dict[str, Any]] = None
) -> Dict[str, List[str]]:
    char_list = []
    for cid, c in characters.items():
        if cid == "narrator":
            continue
        if isinstance(c, dict):
            name = c.get("name", cid)
        else:
            name = cid
        char_list.append(f"- {cid}: {name}")

    if not char_list:
        logger.warning("Nenhuma personagem para extrair aliases.")
        return {}

    char_text = "\n".join(char_list)
    timeout = user_settings.get("ollama_timeout_medium", 120) if user_settings else 120

    prompt = f"""Dado o texto de um livro e a lista de personagens,
identifica todos os termos/aliases que podem referir‑se a cada personagem.

PERSONAGENS:
{char_text}

REGRAS:
1. Inclui nomes próprios, títulos (Sr., Sra., etc.), graus de parentesco (pai, mãe, filho, etc.), profissões, e qualquer outra palavra que a personagem seja chamada no texto.
2. Mantém os termos em minúsculas.
3. Para cada personagem, lista apenas termos que aparecem no texto fornecido.
4. Inclui também termos como "o pai", "a mãe", "o senhor", "a senhora", se aparecerem.
5. Para personagens sem nome próprio (ex: id "vagabundo" ou "jovem_empregado"), inclui termos como "o vagabundo", "um vagabundo", "o homem", etc., se aparecerem no texto.
6. Responde APENAS com JSON válido no formato:
   {{"<character_id_1>": ["nome_proprio", "alcunha"], "<character_id_2>": ["pai", "o pai"], ...}}

TEXTO (primeiros 5000 caracteres para contexto):
{full_text[:5000]}

JSON:"""

    json_schema = {
        "type": "object",
        "additionalProperties": {
            "type": "array",
            "items": {"type": "string"}
        }
    }

    for attempt in range(max_retries + 1):
        try:
            def _make_request():
                return requests.post(ollama_url, json={
                    "model": model_name,
                    "prompt": prompt,
                    "format": json_schema,
                    "stream": False,
                    "keep_alive": -1,
                    "options": {
                        "temperature": 0.1,
                        "top_k": 10,
                        "repeat_penalty": 1.1,
                        "num_ctx": 8192,
                        "num_predict": 8192
                    }
                }, timeout=timeout)

            r = await asyncio.to_thread(_make_request)
            r.raise_for_status()
            raw = r.json().get('response', '').strip()
            raw = raw.replace('```json', '').replace('```', '').strip()
            raw = re.sub(r'[^{]*({.*})', r'\1', raw, flags=re.DOTALL).strip()

            try:
                result = json.loads(raw)
                if isinstance(result, dict):
                    valid_aliases = {}
                    for cid, terms in result.items():
                        if cid in characters and isinstance(terms, list):
                            cleaned = []
                            for t in terms:
                                if isinstance(t, str):
                                    t = t.strip().lower()
                                    if t and len(t) > 1:
                                        cleaned.append(t)
                            if cleaned:
                                valid_aliases[cid] = list(set(cleaned))
                    return valid_aliases
                else:
                    logger.warning(f"Resposta de aliases não é dict: {type(result)}")
                    return {}
            except json.JSONDecodeError:
                if attempt < max_retries:
                    logger.warning(f"JSON de aliases inválido, tentativa {attempt+1}/{max_retries+1}")
                    continue
                return {}
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                logger.warning(f"Timeout na extração de aliases, tentativa {attempt+1}/{max_retries+1}")
                timeout = min(timeout * 1.5, 600)
                continue
            return {}
        except Exception as e:
            logger.warning(f"Erro na extração de aliases: {e}")
            if attempt < max_retries:
                continue
            return {}
    return {}


# ═══════════════════════════════════════════════════════════════════════════
# 7. MAPEADOR DE TERMOS GENÉRICOS
# ═══════════════════════════════════════════════════════════════════════════
class NameMapper:
    def __init__(self, aliases: Optional[Dict[str, List[str]]] = None):
        self.aliases = aliases or {}
        self._reverse_map: Dict[str, str] = {}
        self._build_reverse_map()

    def _build_reverse_map(self):
        self._reverse_map.clear()
        for cid, terms in self.aliases.items():
            for term in terms:
                term_lower = term.lower().strip()
                if term_lower:
                    self._reverse_map[term_lower] = cid

    def update_aliases(self, aliases: Dict[str, List[str]]):
        self.aliases = aliases
        self._build_reverse_map()

    def map_generic(self, term: str, context: str = "") -> str:
        term_lower = term.lower().strip()
        if term_lower in self._reverse_map:
            return self._reverse_map[term_lower]
        for alias, cid in self._reverse_map.items():
            if term_lower in alias or alias in term_lower:
                return cid
        if context:
            for cid, terms in self.aliases.items():
                for alias in terms:
                    if alias in context.lower():
                        return cid
            match = re.search(r'\b([A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][a-záâãàéêíóôõúç]+)\b', context)
            if match:
                name = match.group(1).lower()
                for cid, terms in self.aliases.items():
                    if name in [t.lower() for t in terms] or name in cid:
                        return cid
        if term_lower in ["pai", "mãe", "mae", "pais", "filho", "filha", "irmão", "irmã"]:
            for cid, terms in self.aliases.items():
                for alias in terms:
                    if alias in context.lower():
                        return cid
        return "narrator"

    def resolve_segment_id(self, segment: Dict[str, Any], context_segments: List[Dict[str, Any]] = None) -> str:
        cid = segment.get("character_id", "narrator")
        text = segment.get("text", "")
        if cid in self.aliases or cid == "narrator":
            return cid
        context_text = " ".join([s.get("text", "") for s in (context_segments or [])[-3:]])
        mapped = self.map_generic(cid, context_text + " " + text)
        return mapped if mapped != "narrator" else cid


# ═══════════════════════════════════════════════════════════════════════════
# 8. RESOLVER IDs GENÉRICOS NOS SEGMENTOS
# ═══════════════════════════════════════════════════════════════════════════
def resolve_generic_ids(
    segments: List[Dict[str, Any]],
    characters: Dict[str, Any],
    aliases: Dict[str, List[str]]
) -> List[Dict[str, Any]]:
    if not aliases:
        return segments

    mapper = NameMapper(aliases)
    resolved = []

    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            continue
        new_seg = seg.copy()
        context_segments = segments[max(0, i-3):i]
        new_cid = mapper.resolve_segment_id(new_seg, context_segments)
        if new_cid in characters or new_cid == "narrator":
            new_seg["character_id"] = new_cid
        resolved.append(new_seg)

    return resolved