# core/ollama_analyzer.py
"""
Análise do livro via Ollama (deteção de personagens, segmentação e aliases).
Versão consolidada com prompt original que funcionou com gemma3:27b.
Correções: aumento de num_predict/num_ctx, recuperação de JSON truncado,
verificação de tipo para evitar "str" object does not support item assignment.
"""
import re
import json
import logging
import asyncio
import requests
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

def repair_truncated_json(raw: str) -> str:
    """
    Repara JSON truncado fechando as estruturas abertas na ordem correta (LIFO).
    Estratégia:
    1. Cortar na última vírgula antes do truncamento (remove o segmento incompleto).
    2. Fechar colchetes e chavetas na ordem inversa à abertura, usando uma stack.
    3. Verificar com json.loads; se falhar, fazer fallback para cortar até ao último } ou ].
    """
    if not raw.strip():
        return raw
    raw = raw.strip()

    # ── PASSO 1: Cortar o último item incompleto ───────────────────────────
    # Se o JSON está truncado a meio de um objeto de segmento, o último item
    # provavelmente está incompleto. Cortar na última vírgula que precede
    # um '{' sem fechar é mais seguro do que tentar fechar strings abertas.
    # Encontrar a posição do último '}' completo (heurística: último '}' antes
    # de qualquer conteúdo incompleto).

    # Abordagem: percorrer com stack para saber onde o JSON está completo
    stack = []
    in_string = False
    escape_next = False
    last_complete_pos = -1  # posição do último char onde a estrutura estava "completa"

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
        if ch in ('{', '['):
            stack.append(ch)
        elif ch in ('}', ']'):
            if stack:
                stack.pop()
            if not stack:
                last_complete_pos = i  # estrutura raiz fechada aqui

    # Se a stack está vazia, o JSON já é válido
    if not stack:
        try:
            json.loads(raw)
            return raw
        except json.JSONDecodeError:
            pass  # Há outro problema, continuar com o repair

    # ── PASSO 2: Tentar cortar o último elemento incompleto ───────────────
    # Encontrar a última vírgula fora de strings (antes do truncamento)
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
        # Recalcular a stack para o candidate
        stack2 = []
        in_string2 = False
        escape_next2 = False
        for ch in candidate:
            if escape_next2:
                escape_next2 = False
                continue
            if ch == '\\' and in_string2:
                escape_next2 = True
                continue
            if ch == '"':
                in_string2 = not in_string2
                continue
            if in_string2:
                continue
            if ch in ('{', '['):
                stack2.append(ch)
            elif ch in ('}', ']'):
                if stack2:
                    stack2.pop()

        # Fechar a stack2 na ordem inversa
        closing = ''
        for opener in reversed(stack2):
            closing += '}' if opener == '{' else ']'
        repaired = candidate + closing
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            pass  # falhou, tentar outro método

    # ── PASSO 3: Fechar com a stack original (sem cortar) ─────────────────
    closing = ''
    for opener in reversed(stack):
        closing += '}' if opener == '{' else ']'
    repaired = raw + closing
    try:
        json.loads(repaired)
        return repaired
    except json.JSONDecodeError:
        pass

    # ── PASSO 4: Fallback — varrer de trás para a frente até parsear ──────
    for i in range(len(raw) - 1, -1, -1):
        if raw[i] in ('}', ']'):
            try:
                json.loads(raw[:i + 1])
                return raw[:i + 1]
            except json.JSONDecodeError:
                continue

    return raw

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════
TIMEOUT_SMALL = 120
TIMEOUT_MEDIUM = 300
TIMEOUT_LARGE = 600
MAX_BLOCK_SIZE = 1500  # reduzido para evitar respostas demasiado longas

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
    """
    Carrega o modelo em memória e mantém-no carregado indefinidamente (keep_alive=-1).
    keep_alive deve ser inteiro, não string — "-1s" causa HTTP 400 em versões recentes.
    num_predict=1 evita gerar tokens desnecessários durante o warmup.
    """
    try:
        requests.post(ollama_url, json={
            "model": model_name,
            "prompt": "ok",
            "keep_alive": -1,          # inteiro, não string "-1s"
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

    # ── DETEÇÃO DE LIXO ESTRUTURAL ──────────────────────────────────────────
    # BUG CORRIGIDO: a lista antiga (JUNK_KEYWORDS) continha palavras comuns
    # como "narrator", "text", "pace", "emotion" e procurava-as como SUBSTRING
    # dentro do texto literário do segmento. Isto descartava silenciosamente
    # qualquer frase que contivesse essas substrings por coincidência
    # (ex: "narrador", "contexto", "pretexto", ou qualquer ocorrência de
    # "narrator"/"character_id" mencionada no próprio livro).
    # A nova verificação só marca como lixo fragmentos que estruturalmente
    # parecem ser JSON mal-formado que vazou para o campo "text" (chaves,
    # colchetes, ou pares "chave": valor típicos de JSON), não texto em prosa.
    _json_leak_pattern = re.compile(
        r'"(?:character_id|pause_ms|emotion|pace|text|segments|characters)"\s*:'
    )

    def _is_junk(text: str) -> bool:
        # Só conteúdo estritamente de pontuação/símbolos
        if re.match(r'^[\s\.\,\;\:\!\?\-{}\[\]"\'()]+$', text):
            return True
        # Fragmento de JSON que vazou para o texto (ex: '..."character_id": "narrator"...')
        if _json_leak_pattern.search(text):
            return True
        # Chaves/colchetes nus sem conteúdo literário à volta (não basta conter "{" — tem de ser
        # maioritariamente estrutura JSON)
        symbol_chars = sum(1 for c in text if c in '{}[]":')
        if len(text) > 0 and symbol_chars / len(text) > 0.3:
            return True
        # Caracteres CJK / árabe (idiomas não esperados nesta pipeline PT-PT)
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

    n_in = len(raw_segments)
    n_out = len(result)
    if n_in > 0 and n_out < n_in:
        dropped = n_in - n_out
        ratio = dropped / n_in
        if ratio > 0.2:
            logger.warning(
                f"sanitize_segments: {dropped}/{n_in} segmentos descartados ({ratio:.0%}). "
                f"Se esta percentagem parecer alta, verificar _is_junk()."
            )
        else:
            logger.debug(f"sanitize_segments: {dropped}/{n_in} segmentos descartados ({ratio:.0%}).")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 4. ANÁLISE DE BLOCO (PROMPT ORIGINAL QUE FUNCIONOU)
# ═══════════════════════════════════════════════════════════════════════════
async def analyze_block(
    ollama_url: str,
    model_name: str,
    text: str,
    context: str,
    known_chars: Dict[str, Any],
    aliases: Optional[Dict[str, List[str]]] = None,
    max_retries: int = 2,
    user_settings: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Envia um bloco ao Ollama e retorna personagens + segmentos (Assíncrono).
    """
    # ─── DEFINIR text_length LOGO NO INÍCIO ────────────────────────────────
    text_length = len(text)

    # ─── CONSTRUIR known_list e alias_text ──────────────────────────────────
    known_list = "\n".join([
        f"- {cid}: {c.get('name', cid)}"
        for cid, c in known_chars.items()
        if isinstance(c, dict) and c
    ]) if known_chars else "(nenhuma ainda)"

    alias_text = ""
    if aliases:
        alias_lines = []
        for cid, terms in aliases.items():
            if cid in known_chars:
                alias_lines.append(f"- {cid}: {', '.join(terms)}")
        if alias_lines:
            alias_text = "\nALIASES CONHECIDOS (usa estes para mapear termos genéricos):\n" + "\n".join(alias_lines) + "\n"

    context_block = f"CONTEXTO:\n{context[:500]}...\n\n" if context else ""

    # ─── DEFINIR TIMEOUTS COM BASE NAS DEFINIÇÕES DO UTILIZADOR ────────────
    small = user_settings.get("ollama_timeout_small", TIMEOUT_SMALL) if user_settings else TIMEOUT_SMALL
    medium = user_settings.get("ollama_timeout_medium", TIMEOUT_MEDIUM) if user_settings else TIMEOUT_MEDIUM
    large = user_settings.get("ollama_timeout_large", TIMEOUT_LARGE) if user_settings else TIMEOUT_LARGE

    # ─── PARÂMETROS DINÂMICOS ────────────────────────────────────────────────
    # Estimar tokens do prompt: ~1500 fixos + 1 token por ~4 chars de texto.
    # Adicionar margem de 2× para a resposta JSON.
    # num_ctx mínimo: 8192. Máximo: 32768 (evitar KV cache gigante que causa timeout).
    estimated_prompt_tokens = 1500 + (text_length // 4)
    # A resposta JSON tem ~1 token por char; um bloco de 1500 chars gera ~2000 tokens de JSON
    estimated_response_tokens = max(4096, text_length * 2)
    needed_ctx = estimated_prompt_tokens + estimated_response_tokens

    base_num_ctx     = max(8192, min(needed_ctx, 32768))
    # Arredondar para a potência de 2 mais próxima para eficiência
    for p in (8192, 16384, 32768):
        if base_num_ctx <= p:
            base_num_ctx = p
            break

    base_num_predict = min(estimated_response_tokens, base_num_ctx - estimated_prompt_tokens)
    base_num_predict = max(4096, base_num_predict)

    # Sobrescrever com definições do utilizador se forem maiores
    user_num_predict = user_settings.get("ollama_num_predict", 0) if user_settings else 0
    user_num_ctx     = user_settings.get("ollama_num_ctx",     0) if user_settings else 0
    if user_num_predict > 0:
        base_num_predict = user_num_predict
    if user_num_ctx > 0:
        base_num_ctx = user_num_ctx

    num_predict = base_num_predict
    num_ctx     = base_num_ctx

    # Timeout: escalar com num_ctx (KV cache maior = mais tempo de prefill)
    # Fórmula: small para ≤8k ctx, medium para ≤16k, large para >16k
    if num_ctx <= 8192:
        timeout = small
    elif num_ctx <= 16384:
        timeout = medium
    else:
        timeout = large

    logger.debug(
        f"analyze_block: {text_length} chars → "
        f"num_ctx={num_ctx}, num_predict={num_predict}, timeout={timeout}s"
    )
    prompt = f'''{context_block}/no_think
Analisa este trecho em PT-PT. Identifica TODAS as falas e classifica a EMOÇÃO com base em:

🔹 **PISTAS CONTEXTUAIS** (ordem de prioridade):
1. Verbos dicendi → emoção implícita:
   - "gritou", "berrou", "trovejou" → **angry**
   - "sussurrou", "cochichou", "murmurou" → **whisper**
   - "exclamou", "gritou de alegria" → **joyful**
   - "soluçou", "disse com lágrimas" → **sad**
   - "tremia", "disse com medo" → **fearful**
   - "respondeu", "disse" (neutro) → **neutral** (a menos que haja pista explícita de calma)

2. Pontuação:
   - "!!" → **angry** ou **joyful**
   - "..." → **sad**, **tense** ou **fearful**
   - "?" → **tense** (dúvida, surpresa)

3. Palavras de intensidade:
   - "oh!", "ah!", "uau!" → **joyful**
   - "ai!", "meu Deus!" → **fearful**, **sad** ou **angry**

4. Tom da descrição:
   - "calmamente", "serenamente" → **calm**
   - "nervosamente", "hesitante" → **tense**
   - "com raiva", "irado" → **angry**

⚠️ REGRA MAIS IMPORTANTE SOBRE DESCRIÇÕES DE VOZ:
Para CADA personagem (exceto o narrador), a descrição DEVE ser estritamente sobre a sua VOZ.
NUNCA incluas aspetos físicos (altura, cabelo, olhos) ou relações familiares (esposa, filho).
A descrição deve conter OBRIGATORIAMENTE:
- Género (masculino/feminino)
- Idade (aproximada: "jovem", "30-40", "idoso")
- Tom de voz (ex: "grave", "aguda", "suave", "autoritária")
- Sotaque: "Português de Portugal" (sempre)

Exemplos CORRETOS:
✅ "Voz masculina, 50 anos, grave e autoritária, sotaque de Lisboa."
✅ "Voz feminina, jovem (16-18), aguda e ansiosa, sotaque do Porto."
✅ "Voz masculina, madura, calma e serena, sotaque de Portugal continental."

Exemplos INCORRETOS (NUNCA usar):
❌ "Homem de uns cinquenta anos." (falta voz)
❌ "Mulher com convicção." (não é voz)
❌ "Jovem alto e franzino." (físico, não voz)

REGRAS:
- Se não houver pistas claras de emoção → **neutral**
- **NUNCA** atribuas uma emoção sem pista; usa `neutral` apenas como último recurso.
Emoções permitidas: neutral, calm, tense, joyful, sad, angry, fearful, whisper.

---

REGRAS GERAIS:
1. **Narrador**: usa `"character_id": "narrator"` APENAS para:
   - Texto descritivo (acções, cenários, movimentos)
   - Pensamentos do narrador
   - Títulos, cabeçalhos, epígrafes, notas
   - Texto que não é fala de ninguém

2. **Diálogo (fala de personagens)**:
   - Sempre que houver aspas (« » ou " ") ou travessão (—) a indicar fala.
   - Identifica a personagem pelo seu **nome próprio** (ex: "asa_griffiths").
   - Se o falante for referido como "o pai", "a mãe", etc., usa o alias correspondente.
   - **IMPORTANTE**: Se uma personagem fala mas NÃO tem nome próprio (ex: "um jovem empregado", "um vagabundo", "uma senhora", "um transeunte"), CRIA um ID descritivo em minúsculas com underscores (ex: "jovem_empregado", "vagabundo", "senhora", "transeunte").
   - **NUNCA** atribuas uma fala a `"narrator"` se houver aspas ou travessão a indicar que alguém está a falar.

3. **Nomes próprios**: usa sempre minúsculas e underscores (ex: "clyde_griffiths").

4. **Aliases**: sempre que vires "pai", "o pai", usa "asa_griffiths"; "mãe", "a mãe" → "elvira_griffiths".

5. **Emoção**: usa APENAS: neutral, calm, tense, joyful, sad, angry, fearful, whisper.

6. **Títulos e cabeçalhos**: devem ser marcados como `"character_id": "narrator"` e `"emotion": "neutral"`.

PERSONAGENS JÁ CONHECIDAS (usa estes IDs sempre que possível):
{known_list}

{alias_text}

EXEMPLOS:
- Texto: — Vamos cantar o hino, disse o pai.
  Saída: {{"character_id": "asa_griffiths", "emotion": "calm", "pace": 1.0, "pause_ms": 300}}

- Texto: "Vejo estas pessoas por aqui quase todas as noites", disse um jovem empregado.
  Saída: {{"character_id": "jovem_empregado", "emotion": "neutral", "pace": 1.0, "pause_ms": 200}}

- Texto: "Aquele miúdo mais velho não quer estar aqui", observou um vagabundo.
  Saída: {{"character_id": "vagabundo", "emotion": "neutral", "pace": 1.0, "pause_ms": 200}}

- Texto: "É, acho que sim", concordou o outro transeunte.
  Saída: {{"character_id": "transeunte", "emotion": "neutral", "pace": 1.0, "pause_ms": 150}}

- Texto: Capítulo 1 — O Início
  Saída: {{"character_id": "narrator", "emotion": "neutral", "pace": 1.0, "pause_ms": 800}}

TEXTO A ANALISAR:
""" {text} """

Responde APENAS com JSON válido, sem texto extra.
JSON: {{"characters": {{"narrator": {{"name": "Narrador", "type": "narrator", "description": "Voz masculina madura, português de Portugal"}}}}, "segments": [{{"text": "...", "character_id": "narrator", "emotion": "neutral", "pace": 1.0, "pause_ms": 0}}]}}'''

    # ─── SCHEMA JSON ──────────────────────────────────────────────────────────
    json_schema = {
        "type": "object",
        "properties": {
            "characters": {"type": "object"},
            "segments": {"type": "array"}
        },
        "required": ["characters", "segments"]
    }

    # ─── LOOP DE TENTATIVAS ──────────────────────────────────────────────────
    for attempt in range(max_retries + 1):
        try:
            def _make_request():
                return requests.post(ollama_url, json={
                    "model": model_name,
                    "prompt": prompt,
                    "format": json_schema,
                    "stream": False,
                    "keep_alive": -1,      # mantém o modelo em memória após a resposta
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

            # ── LOG DA RESPOSTA BRUTA (para diagnóstico) ──────────────────────
            logger.debug(f"Resposta bruta (início): {raw[:200]}...")

            # ── LIMPEZA E EXTRAÇÃO DO JSON ────────────────────────────────────
            raw = raw.replace('```json', '').replace('```', '').strip()

            # Tentar extrair apenas o JSON válido com regex
            match = re.search(r'(\{.*\})', raw, re.DOTALL)
            if match:
                raw = match.group(1)
            else:
                # Fallback: remover caracteres não permitidos em JSON
                raw = re.sub(r'[^\{\}\[\]":,0-9a-zA-Z_\-\. ]', '', raw)

            logger.debug(f"Resposta após limpeza: {raw[:200]}...")

            # ── VERIFICAR SE O JSON ESTÁ TRUNCADO ─────────────────────────────
            open_braces = raw.count('{')
            close_braces = raw.count('}')
            open_brackets = raw.count('[')
            close_brackets = raw.count(']')

            is_truncated = False
            if open_braces > close_braces or open_brackets > close_brackets:
                is_truncated = True
                logger.warning(
                    f"JSON truncado: {open_braces}x'{{' vs {close_braces}x'}}', "
                    f"{open_brackets}x'[' vs {close_brackets}x']' — "
                    f"tamanho={len(raw)} chars — "
                    f"fim: ...{raw[-120:]!r}"
                )

            # ── RECUPERAR JSON TRUNCADO (FECHAR CHAVES/COLCHETES) ─────────────
            if is_truncated:
                raw = repair_truncated_json(raw)   # <-- chamada direta (mesmo ficheiro)
                logger.debug(f"JSON reparado: {raw[:200]}...")

            try:
                result = json.loads(raw)

                if "characters" in result:
                    valid_chars = {}
                    for cid, cdata in result["characters"].items():
                        cid_str = str(cid).strip().lower().replace(" ", "_")
                        # Verificação rigorosa: cdata deve ser dict
                        if isinstance(cdata, dict) and cdata:
                            desc = cdata.get("description", "")
                            if not desc.strip() or "voz" not in desc.lower():
                                name = cdata.get("name", cid_str)
                                if "jovem" in name.lower() or "rapaz" in name.lower():
                                    desc = f"Voz jovem de {name}, português de Portugal, tom neutro"
                                elif "velho" in name.lower() or "vagabundo" in name.lower():
                                    desc = f"Voz madura de {name}, português de Portugal, tom neutro"
                                elif "senhora" in name.lower() or "mulher" in name.lower() or "mãe" in name.lower():
                                    desc = f"Voz feminina de {name}, português de Portugal, tom neutro"
                                elif "senhor" in name.lower() or "homem" in name.lower() or "pai" in name.lower():
                                    desc = f"Voz masculina de {name}, português de Portugal, tom neutro"
                                else:
                                    desc = f"Voz de {name}, português de Portugal, tom neutro"
                                cdata["description"] = desc
                            valid_chars[cid_str] = cdata
                        else:
                            # Se cdata não for dict, criar um dict básico
                            logger.warning(f"cdata para {cid} não é dict: {type(cdata)}. A criar descrição padrão.")
                            valid_chars[cid_str] = {
                                "name": cid_str,
                                "type": "character",
                                "description": f"Voz de {cid_str}, português de Portugal, tom neutro"
                            }
                    result["characters"] = valid_chars

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
                logger.warning(f"  início: {raw[:200]!r}")
                logger.warning(f"  fim:    {raw[-200:]!r}")
                if attempt < max_retries:
                    # Aumentar num_predict e num_ctx para a próxima tentativa
                    num_predict = min(num_predict * 2, 131072)
                    num_ctx = min(num_ctx * 2, 65536)
                    logger.warning(f"A aumentar num_predict para {num_predict} e num_ctx para {num_ctx} e a tentar novamente")
                    continue
                return {"characters": {}, "segments": []}

        except requests.exceptions.Timeout:
            if attempt < max_retries:
                logger.warning(f"Timeout ({text_length} chars), tentativa {attempt + 1}/{max_retries + 1}")
                timeout = min(timeout * 1.5, 600)
                # Aumentar também os limites de tokens
                num_predict = min(num_predict * 2, 32768)
                num_ctx = min(num_ctx * 2, 16384)
                continue
            return {"characters": {}, "segments": []}

        except Exception as e:
            logger.warning(f"Erro Ollama: {e}")
            if attempt < max_retries:
                continue
            return {"characters": {}, "segments": []}

    return {"characters": {}, "segments": []}


# ═══════════════════════════════════════════════════════════════════════════
# 5. EXTRAÇÃO DE ALIASES
# ═══════════════════════════════════════════════════════════════════════════
async def extract_aliases(
    ollama_url: str,
    model_name: str,
    full_text: str,
    characters: Dict[str, dict],
    max_retries: int = 2,
    user_settings: Optional[Dict[str, Any]] = None
) -> Dict[str, List[str]]:
    """
    Extrai aliases (apelidos/termos alternativos) para cada personagem.
    Agora com timeout baseado nas definições do utilizador.
    """
    char_list = []
    for cid, c in characters.items():
        if cid == "narrator":
            continue
        # Verificar se c é dict antes de aceder
        if isinstance(c, dict):
            name = c.get("name", cid)
        else:
            name = cid
        char_list.append(f"- {cid}: {name}")

    if not char_list:
        logger.warning("Nenhuma personagem para extrair aliases.")
        return {}

    char_text = "\n".join(char_list)

    # ── DEFINIR TIMEOUT COM BASE NAS DEFINIÇÕES DO UTILIZADOR ──────────────
    timeout = user_settings.get("ollama_timeout_medium", 120) if user_settings else 120

    prompt = f"""Dado o texto de um livro e a lista de personagens,
identifica todos os termos/aliases que podem referir‑se a cada personagem.

PERSONAGENS:
{char_text}

REGRAS:
1. Inclui nomes próprios, títulos (Sr., Sra., etc.), graus de parentesco (pai, mãe, filho, etc.), profissões, e qualquer outra palavra que a personagem seja chamada no texto.
2. Mantém os termos em minúsculas.
3. Para cada personagem, lista apenas termos que aparecem no texto fornecido.
4. Inclui também termos como "o pai", "a mãe", "o senhor", "a senhora", "o vagabundo" se aparecerem.
5. Para personagens sem nome próprio (ex: "vagabundo", "jovem_empregado"), inclui termos como "o vagabundo", "um vagabundo", "o homem", etc., se aparecerem no texto.
6. Responde APENAS com JSON válido no formato:
   {{"asa_griffiths": ["pai", "o pai", "asa"], "elvira_griffiths": ["mãe", "a mãe", ...], "vagabundo": ["o vagabundo", "um vagabundo", "homem"]}}

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

    # ── LOOP DE TENTATIVAS ──────────────────────────────────────────────────
    for attempt in range(max_retries + 1):
        try:
            def _make_request():
                return requests.post(ollama_url, json={
                    "model": model_name,
                    "prompt": prompt,
                    "format": json_schema,
                    "stream": False,
                    "keep_alive": -1,      # mantém o modelo em memória após a resposta
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
                    logger.warning(f"JSON de aliases inválido, tentativa {attempt + 1}/{max_retries + 1}")
                    continue
                return {}

        except requests.exceptions.Timeout:
            if attempt < max_retries:
                logger.warning(f"Timeout na extração de aliases, tentativa {attempt + 1}/{max_retries + 1}")
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
# 6. MAPEADOR DE TERMOS GENÉRICOS
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
        # 1. correspondência exata
        if term_lower in self._reverse_map:
            return self._reverse_map[term_lower]
        # 2. correspondência parcial
        for alias, cid in self._reverse_map.items():
            if term_lower in alias or alias in term_lower:
                return cid
        # 3. procurar no contexto
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
        # 4. termos de parentesco
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
# 7. RESOLVER IDs GENÉRICOS NOS SEGMENTOS
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