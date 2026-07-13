# core/ollama_analyzer.py
"""
Análise do livro via Ollama (deteção de personagens, segmentação e aliases).
Versão consolidada com prompt original que funcionou com gemma3:27b.
Correções: aumento de num_predict/num_ctx, recuperação de JSON truncado,
verificação de tipo para evitar "str" object does not support item assignment.
"""
import re
import time
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
        # SE O PARÁGRAFO TEM DIÁLOGO (ASPAS OU TRAVESSÃO), NÃO O DIVIDIR!
        # Forçar o modelo a ver o parágrafo inteiro para entender o contexto.
        has_dialogue = bool(re.search(r'["«»—]', para))
        
        if len(para) > max_chars and not has_dialogue:
            # Só divide parágrafos longos que NÃO sejam diálogo
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
            # Parágrafos de diálogo (mesmo que gigantes) vão inteiros ou agrupados
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
# 4. FASE 1 — DESCOBERTA DE PERSONAGENS (varredura do livro completo)
# ═══════════════════════════════════════════════════════════════════════════
async def discover_characters(
    ollama_url: str,
    model_name: str,
    full_text: str,
    max_retries: int = 2,
    user_settings: Optional[Dict[str, Any]] = None,
    progress_callback=None
) -> Dict[str, Any]:
    """
    Fase 1 do pipeline de dois passos.

    Varre o texto completo do livro em blocos grandes (até 12 000 chars)
    e pede ao modelo APENAS que identifique:
      - Quais personagens existem
      - Quais têm discurso direto genuíno (falas, diálogos, cartas)
      - Descrição de voz para cada uma

    NÃO faz segmentação — isso fica para analyze_block (Fase 2).
    Deste modo o modelo não está a fazer duas tarefas complexas ao mesmo tempo,
    e o elenco descoberto é passado completo a cada bloco de segmentação.

    Devolve um dict de personagens no mesmo formato que analyze_block usa,
    com "narrator" já incluído.
    """
    timeout = user_settings.get("ollama_timeout_large", TIMEOUT_LARGE) if user_settings else TIMEOUT_LARGE

    # Dividir o texto em blocos de 12 000 chars com sobreposição de 500 chars
    # para não perder personagens que aparecem no limite entre blocos.
    chunk_size = 12000
    overlap   = 500
    chunks = []
    start = 0
    while start < len(full_text):
        end = min(start + chunk_size, len(full_text))
        chunks.append(full_text[start:end])
        if end == len(full_text):
            break
        start = end - overlap

    logger.info(f"🔎 Fase 1 — Descoberta de personagens em {len(chunks)} blocos...")
    start_phase1 = time.time()
    logger.info(f"🔎 Fase 1 — Início: {time.strftime('%H:%M:%S')}")

    generic_terms_seen: set = set()  # termos de papel/parentesco vistos mas não persistidos

    all_characters: Dict[str, Any] = {
        "narrator": {
            "name": "Narrador",
            "type": "narrator",
            "description": "Voz masculina madura, português de Portugal"
        }
    }

    json_schema = {
        "type": "object",
        "properties": {
            "characters": {"type": "object"}
        },
        "required": ["characters"]
    }

    # Padrões que indicam que o próprio modelo admite, na descrição, que a
    # personagem NÃO tem discurso direto — apesar da instrução do prompt
    # pedir explicitamente para excluir personagens apenas mencionadas.
    # Isto é uma rede de segurança independente da obediência do modelo à
    # instrução, e é uma verificação a nível de língua (não específica de
    # nenhum livro), pelo que mantém o sistema universal.
    _no_direct_speech_pattern = re.compile(
        r'sem\s+di[aá]logo\s+direto|sem\s+discurso\s+direto|sem\s+fala\s+direta|'
        r'apenas\s+mencionad[oa]|mencionad[oa]\s+mas\s+sem|n[ãa]o\s+tem\s+fala|'
        r'sem\s+falas?\s+(?:atribu[ií]das|registadas)',
        re.IGNORECASE
    )

    # ═══════════════════════════════════════════════════════════════════════
    # TERMOS GENÉRICOS DE PAPEL/PARENTESCO (nível de LÍNGUA, não de livro)
    # ═══════════════════════════════════════════════════════════════════════
    # BUG CORRIGIDO (causa raiz de "pai"/"mae_clyde" a absorver falas de
    # dezenas de pessoas diferentes ao longo do livro todo):
    #
    # A Fase 1 varre o livro em ~165 blocos sequenciais. Quando um bloco não
    # revela o nome próprio de quem fala (ex: "disse o pai"), o modelo cria
    # um ID genérico como "pai". Até aqui está correto — mas esse ID entra
    # na lista "PERSONAGENS JÁ CONHECIDAS" passada a TODOS os blocos da Fase 2
    # (incluindo blocos centenas de milhares de caracteres à frente, sobre
    # famílias e personagens completamente diferentes). Como o prompt da
    # Fase 2 diz "usa estes IDs sempre que possível", QUALQUER figura paternal
    # não identificada no resto do livro — o pai de outra personagem, um juiz,
    # um reverendo — acaba a ser rotulada com o MESMO "pai", fundindo pessoas
    # sem relação nenhuma entre si sob uma única identidade.
    #
    # Um nome próprio (ex: "clyde_griffiths") refere-se sempre à mesma pessoa
    # em qualquer ponto do livro — persistir esse ID entre blocos é seguro e
    # é precisamente o que resolve a fragmentação de personagens. Mas um
    # termo de PAPEL/PARENTESCO genérico ("pai", "mãe", "guia", "guarda") não
    # tem essa propriedade: o mesmo termo aplica-se legitimamente a pessoas
    # diferentes em pontos diferentes de um romance longo com várias famílias.
    #
    # Solução: candidatos cujo nome é um substantivo comum de papel/parentesco
    # (lista de língua portuguesa, não de nenhum livro específico) NÃO entram
    # na lista persistente reutilizada entre blocos — ficam de fora do "elenco
    # fixo" da Fase 2, para que cada bloco possa tratá-los localmente em vez
    # de os fundir globalmente. Isto evita a conflação, ao custo de tais falas
    # (quando o livro nunca revela o nome próprio) caírem no narrador em vez
    # de ficarem com uma etiqueta de papel persistente — um resultado mais
    # seguro do que atribuir falas de estranhos à mesma "pessoa" fictícia.
    GENERIC_ROLE_TERMS_PT = {
        "pai", "mae", "mãe", "tio", "tia", "avo", "avó", "avô",
        "marido", "esposa", "mulher", "homem", "senhor", "senhora",
        "menina", "menino", "moça", "moço", "rapaz", "rapariga",
        "dona", "guia", "guarda", "vizinho", "vizinha",
        "medico", "médico", "médica", "enfermeira", "enfermeiro",
        "juiz", "advogado", "procurador", "policia", "polícia",
        "soldado", "capitao", "capitão", "padre", "reverendo",
        "professor", "professora", "estranho", "estranha",
        "velho", "velha", "jovem", "criada", "empregado", "empregada",
        "cocheiro", "carteiro", "caixa", "vendedor", "vendedora",
        "meritissimo", "meritíssimo", "testemunha", "vagabundo",
    }

    _relational_compound_pattern = re.compile(
        r'^(mãe|mae|pai|irm[ãa]o?|tio|tia|av[oó]|filho|filha|esposa|marido|'
        r'sobrinho|sobrinha|primo|prima|cunhad[oa]|sogr[oa]|nor[ao])\s+d[eo]\s+\w+',
        re.IGNORECASE
    )
    # Preposição possessiva/relacional em qualquer parte do nome
    _possessive_preposition_pattern = re.compile(r'\bd[eoa]s?\b', re.IGNORECASE)
    _role_word_pattern = re.compile(
        r'\b(' + '|'.join(re.escape(t) for t in sorted(GENERIC_ROLE_TERMS_PT, key=len, reverse=True)) + r')\b',
        re.IGNORECASE
    )

    def _is_generic_role_term(cdata: Dict[str, Any], cid: str) -> bool:
        """Verifica se o candidato é um termo de papel/parentesco genérico
        (não um nome próprio), usando o campo 'name' (mais fiável que o id,
        que já vem normalizado com underscores).

        Cobre três padrões:
        1. Termo isolado exato (ex: "pai", "a mãe", "o guia").
        2. Descritor relacional composto na ordem "papel + de/do + X"
           (ex: "Mãe de Clyde") — originou o bug "mae_clyde".
        3. BUG CORRIGIDO (2ª ronda): descritor relacional na ordem INVERSA,
           "X + de/do/da + papel" (ex: "Mulher do Pai", "Marido da Mulher
           do Quarto 529") — o padrão 2 só cobria uma direção gramatical;
           esta variante continuou a causar exatamente a mesma conflação
           de personagens sem relação entre si (ex: "mulher_do_pai" corrigido
           pela revisão para 19 personagens diferentes: Roberta Howard,
           Gilbert Griffiths, Simeon Dinsmore, etc.).
           Heurística: a MERA co-ocorrência de uma preposição possessiva
           (de/do/da/dos/das) com uma palavra de papel/parentesco em
           qualquer ordem é o sinal estrutural de que se trata de uma
           referência relacional, não de um nome próprio autónomo — mesmo
           sem enumerar cada ordem/frase possível.
        """
        name = str(cdata.get("name", cid)).strip().lower()
        name_stripped = re.sub(r'^(o|a|um|uma|seu|sua|meu|minha)\s+', '', name)
        if name_stripped in GENERIC_ROLE_TERMS_PT or cid in GENERIC_ROLE_TERMS_PT:
            return True
        if _relational_compound_pattern.match(name_stripped):
            return True
        # Padrão 3: preposição possessiva + palavra de papel, em qualquer ordem
        if _possessive_preposition_pattern.search(name_stripped) and _role_word_pattern.search(name_stripped):
            return True
        return False

    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(i / len(chunks), f"Fase 1: bloco {i+1}/{len(chunks)}")

        known_so_far = "\n".join(
            f"- {cid}: {c.get('name', cid)}"
            for cid, c in all_characters.items()
            if isinstance(c, dict)
        ) if all_characters else "(nenhuma ainda)"

        prompt = f"""/no_think
Lê este excerto de um livro e lista APENAS as personagens que têm DISCURSO DIRETO (falas, diálogos, cartas transcritas).

PERSONAGENS JÁ IDENTIFICADAS (não as repitas, só adiciona novas):
{known_so_far}

REGRAS:
1. Inclui APENAS personagens com fala própria (aspas « » " " ou travessão —). NÃO incluas personagens apenas mencionadas.
2. Usa IDs em minúsculas com underscores (ex: "joao_silva", "pai", "jovem_empregado").
3. A "description" DEVE descrever a VOZ: género, idade aproximada, tom, sotaque PT-PT.
4. Para o narrador, usa o ID "narrator" — mas só se tiver fala.
5. NÃO faças segmentação. Devolve APENAS o JSON de personagens.

TEXTO:
{chunk}

Responde APENAS com JSON válido:
{{"characters": {{"<id>": {{"name": "<Nome>", "type": "personagem", "description": "Voz ..."}}}}}}"""

        for attempt in range(max_retries + 1):
            try:
                def _req(p=prompt):
                    return requests.post(ollama_url, json={
                        "model": model_name,
                        "prompt": p,
                        "format": json_schema,
                        "stream": False,
                        "keep_alive": -1,
                        "options": {
                            "temperature": 0.1,
                            "top_k": 20,
                            "num_ctx": 16384,
                            "num_predict": 4096,
                        }
                    }, timeout=timeout)

                r = await asyncio.to_thread(_req)
                r.raise_for_status()
                raw = r.json().get("response", "").strip()
                raw = raw.replace("```json", "").replace("```", "").strip()
                m = re.search(r'(\{.*\})', raw, re.DOTALL)
                if m:
                    raw = m.group(1)

                data = json.loads(raw)
                chars = data.get("characters", {})
                if not isinstance(chars, dict):
                    break

                for cid, cdata in chars.items():
                    cid = str(cid).strip().lower().replace(" ", "_")
                    if not cid or cid == "narrator":
                        continue
                    if not isinstance(cdata, dict) or not cdata:
                        continue

                    # BUG CORRIGIDO: apesar do prompt instruir explicitamente
                    # "NÃO incluas personagens apenas mencionadas", o modelo
                    # por vezes inclui-as na mesma, confessando isso na própria
                    # descrição (ex: "(mencionado mas sem diálogo direto)").
                    # Rejeitar estes casos aqui, de forma programática.
                    desc_check = str(cdata.get("description", ""))  # <-- FORÇAR STRING
                    if _no_direct_speech_pattern.search(desc_check):
                        logger.debug(f"  Bloco {i+1}: candidato '{cid}' rejeitado (descrição admite ausência de discurso direto).")
                        continue

                    # BUG CORRIGIDO (causa raiz pai/mae_clyde): termos genéricos
                    # de papel/parentesco não entram na lista persistente —
                    # ver explicação completa acima de GENERIC_ROLE_TERMS_PT.
                    if _is_generic_role_term(cdata, cid):
                        generic_terms_seen.add(cid)
                        continue

                    # Manter a descrição mais rica se a personagem já existe
                    if cid in all_characters:
                        existing_desc = str(all_characters[cid].get("description", ""))  # <-- FORÇAR STRING
                        new_desc = str(cdata.get("description", ""))                     # <-- FORÇAR STRING
                        if len(new_desc) > len(existing_desc):
                            all_characters[cid]["description"] = new_desc
                    else:
                        # Garantir descrição de voz mínima
                        desc = str(cdata.get("description", ""))  # <-- FORÇAR STRING
                        if not desc or "voz" not in desc.lower():
                            name = str(cdata.get("name", cid))    # <-- FORÇAR STRING (por segurança)
                            cdata["description"] = f"Voz neutra, português de Portugal, tom neutro. ({name})"
                        all_characters[cid] = cdata

                logger.debug(f"  Bloco {i+1}: +{len(chars)} personagens, total={len(all_characters)-1}")
                break  # sucesso — avançar para o próximo chunk

            except (json.JSONDecodeError, KeyError):
                if attempt < max_retries:
                    continue
            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    continue
                logger.warning(f"  Timeout na descoberta de personagens (bloco {i+1})")
            except Exception as e:
                logger.warning(f"  Erro na descoberta (bloco {i+1}): {e}")
                if attempt < max_retries:
                    continue
                break

    n = len(all_characters) - 1  # excluir narrator
    logger.info(f"✅ Fase 1 (bruto) concluída: {n} personagem(s) candidata(s) identificada(s).")

    # ═══════════════════════════════════════════════════════════════════════
    # CONSOLIDAÇÃO FINAL — necessária para livros longos (muitos blocos)
    # ═══════════════════════════════════════════════════════════════════════
    # Em livros grandes (400k+ chars → ~30+ blocos de descoberta processados
    # sequencialmente), o modelo pode não reconhecer que uma personagem já
    # descoberta num bloco anterior está a ser referida de forma diferente
    # num bloco muito posterior (nome incompleto, alcunha, erro de grafia),
    # criando um character_id duplicado para a mesma pessoa. Isto fragmentaria
    # a voz dessa personagem em dois IDs distintos durante toda a Fase 2.
    #
    # Correção em duas etapas, com custo O(1) independente do tamanho do livro:
    #   1. Clustering programático barato (sem chamada ao Ollama): agrupa
    #      candidatos cujo nome de um está contido no nome do outro.
    #   2. UMA única chamada final ao Ollama com a lista já pré-agrupada,
    #      para resolver ambiguidades que a heurística de substring não
    #      consegue (nomes muito diferentes para a mesma pessoa).
    all_characters = _cluster_characters_by_name(all_characters)
    all_characters = await _consolidate_characters_via_llm(
        all_characters, ollama_url, model_name,
        timeout=user_settings.get("ollama_timeout_medium", TIMEOUT_MEDIUM) if user_settings else TIMEOUT_MEDIUM
    )

    n_final = len(all_characters) - 1
    logger.info(f"✅ Fase 1 (consolidada) concluída: {n_final} personagem(s) final(is).")
    end_phase1 = time.time()
    duration = end_phase1 - start_phase1
    logger.info(f"✅ Fase 1 concluída em {duration:.1f}s ({duration/60:.1f}min)")

    for cid, c in all_characters.items():
        if cid != "narrator" and isinstance(c, dict):
            logger.info(f"   - {cid}: {c.get('name', cid)}")

    if generic_terms_seen:
        logger.info(
            f"ℹ️ {len(generic_terms_seen)} termo(s) genérico(s) de papel/parentesco vistos mas "
            f"NÃO persistidos entre blocos (evita conflação de pessoas diferentes): "
            f"{sorted(generic_terms_seen)}"
        )

    return all_characters


def _cluster_characters_by_name(characters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Agrupa candidatos a personagem cujo nome/ID é uma variante trivial de outro
    (ex: "clyde" e "clyde_griffiths" — um nome próprio contido no outro).
    Mantém o ID mais específico (mais longo) como canónico e funde a descrição
    mais rica. Não usa o Ollama — é uma heurística barata, sempre segura de
    correr primeiro para reduzir o trabalho da consolidação por LLM a seguir.

    BUG CORRIGIDO: candidatos como "Duncan McMillan" e "Reverendo McMillan"
    (a mesma pessoa, confirmado no texto por uma auto-apresentação: "O meu
    nome é Duncan McMillan") não eram fundidos porque só partilham o apelido
    — nenhum nome está inteiramente contido no outro (falha do teste de
    substring). Adicionado um segundo critério: se dois candidatos partilham
    a MESMA última palavra (apelido) e um deles é "título/papel + apelido"
    (ex: "Reverendo McMillan") enquanto o outro tem um primeiro nome próprio
    diferente antes do mesmo apelido (ex: "Duncan McMillan"), são a mesma
    pessoa — um título formal a referir-se a alguém já apresentado pelo nome
    próprio antes.
    """
    TITLE_WORDS_PT = {
        "reverendo", "reverenda", "senhor", "senhora", "sr", "sra", "dr", "dra",
        "doutor", "doutora", "capitao", "capitão", "sargento", "coronel",
        "general", "professor", "professora", "juiz", "juíza", "padre",
        "tenente", "major", "governador", "presidente", "xerife",
    }

    entries = [(cid, c) for cid, c in characters.items() if cid != "narrator" and isinstance(c, dict)]
    # Ordenar por comprimento do nome, mais longo primeiro, para que nomes
    # mais específicos "absorvam" nomes mais curtos contidos neles.
    entries.sort(key=lambda kv: -len(kv[1].get("name", kv[0])))

    canonical: Dict[str, Any] = {}
    rename_map: Dict[str, str] = {}

    def _last_token(name: str) -> str:
        parts = name.strip().lower().split()
        return parts[-1] if parts else ""

    def _first_token(name: str) -> str:
        parts = name.strip().lower().split()
        return parts[0] if parts else ""

    for cid, cdata in entries:
        name_norm = cdata.get("name", cid).strip().lower()
        merged_into = None
        for existing_cid, existing_data in canonical.items():
            existing_name_norm = existing_data.get("name", existing_cid).strip().lower()
            if not name_norm or not existing_name_norm:
                continue
            # Critério 1: um nome inteiramente contido no outro
            if len(name_norm) >= 3 and (name_norm in existing_name_norm or existing_name_norm in name_norm):
                merged_into = existing_cid
                break
            # Critério 2: mesmo apelido (última palavra) + um dos dois é
            # "título + apelido" — provável referência formal à mesma pessoa.
            last_a, last_b = _last_token(name_norm), _last_token(existing_name_norm)
            if last_a and last_a == last_b and len(last_a) >= 4:
                first_a, first_b = _first_token(name_norm), _first_token(existing_name_norm)
                a_is_title = first_a in TITLE_WORDS_PT
                b_is_title = first_b in TITLE_WORDS_PT
                # Um é título+apelido, o outro é nome-próprio+apelido (não título)
                if a_is_title != b_is_title:
                    merged_into = existing_cid
                    break
        if merged_into:
            rename_map[cid] = merged_into
            # Manter a descrição mais rica entre as duas
            old_desc = cdata.get("description", "")
            canon_desc = canonical[merged_into].get("description", "")
            if len(old_desc) > len(canon_desc):
                canonical[merged_into]["description"] = old_desc
        else:
            canonical[cid] = cdata

    if rename_map:
        logger.info(f"   🧹 Clustering programático: {len(rename_map)} candidato(s) fundido(s) por nome semelhante.")

    canonical["narrator"] = characters.get("narrator", {
        "name": "Narrador", "type": "narrator",
        "description": "Voz masculina madura, português de Portugal"
    })
    return canonical


async def _consolidate_characters_via_llm(
    characters: Dict[str, Any],
    ollama_url: str,
    model_name: str,
    timeout: int = 300
) -> Dict[str, Any]:
    """
    Passagem final de consolidação: envia a lista já pré-agrupada ao Ollama
    e pede para identificar quaisquer pares restantes que sejam a MESMA
    personagem (nomes muito diferentes para a mesma pessoa — alcunhas,
    títulos, etc. — que o clustering por substring não apanha).

    Custo constante: uma única chamada, independentemente do tamanho do livro,
    porque só depende do número de personagens candidatas (tipicamente
    dezenas, não milhares).
    """
    candidates = {cid: c for cid, c in characters.items() if cid != "narrator" and isinstance(c, dict)}
    if len(candidates) <= 1:
        return characters  # nada a consolidar

    candidate_list = "\n".join(
        f'- "{cid}": nome="{c.get("name", cid)}", descrição="{c.get("description", "")}"'
        for cid, c in candidates.items()
    )

    prompt = f"""/no_think
Aqui está uma lista de personagens candidatas, extraídas de diferentes partes de um livro.
Algumas podem ser DUPLICADAS da mesma pessoa (ex: alcunha vs nome completo, erro de grafia, título vs nome).

LISTA:
{candidate_list}

Identifica APENAS os grupos que são claramente a MESMA pessoa. Se não tiveres a certeza, NÃO os agrupes.
Responde APENAS com JSON no formato:
{{"merges": [{{"keep": "<id_canónico_a_manter>", "remove": ["<id_duplicado_1>", "<id_duplicado_2>"]}}]}}
Se não houver duplicados, responde: {{"merges": []}}"""

    json_schema = {
        "type": "object",
        "properties": {"merges": {"type": "array"}},
        "required": ["merges"]
    }

    try:
        def _req():
            return requests.post(ollama_url, json={
                "model": model_name,
                "prompt": prompt,
                "format": json_schema,
                "stream": False,
                "keep_alive": -1,
                "options": {"temperature": 0.0, "num_ctx": 8192, "num_predict": 2048}
            }, timeout=timeout)

        r = await asyncio.to_thread(_req)
        r.raise_for_status()
        raw = r.json().get("response", "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        m = re.search(r'(\{.*\})', raw, re.DOTALL)
        if m:
            raw = m.group(1)
        data = json.loads(raw)
        merges = data.get("merges", [])
        if not isinstance(merges, list):
            return characters

        for merge in merges:
            if not isinstance(merge, dict):
                continue
            keep = str(merge.get("keep", "")).strip().lower().replace(" ", "_")
            remove_ids = merge.get("remove", [])
            if keep not in characters or not isinstance(remove_ids, list):
                continue
            for rid in remove_ids:
                rid = str(rid).strip().lower().replace(" ", "_")
                if rid == keep or rid not in characters:
                    continue
                old_desc = characters[rid].get("description", "")
                keep_desc = characters[keep].get("description", "")
                if len(old_desc) > len(keep_desc):
                    characters[keep]["description"] = old_desc
                del characters[rid]
                logger.info(f"   🔗 Consolidação LLM: '{rid}' fundido em '{keep}'.")

    except Exception as e:
        logger.warning(f"⚠️ Consolidação final de personagens falhou (não crítico, mantendo lista bruta): {e}")

    return characters



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
    Envia um bloco ao Ollama e retorna uma lista de segmentos finos,
    onde cada segmento é uma unidade indivisível de fala/narração.
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
    estimated_prompt_tokens = 1500 + (text_length // 4)
    estimated_response_tokens = max(4096, text_length * 2)
    needed_ctx = estimated_prompt_tokens + estimated_response_tokens

    base_num_ctx = max(8192, min(needed_ctx, 12288))
    for p in (8192, 16384, 32768):
        if base_num_ctx <= p:
            base_num_ctx = p
            break

    base_num_predict = min(estimated_response_tokens, base_num_ctx - estimated_prompt_tokens)
    base_num_predict = max(4096, base_num_predict)

    user_num_predict = user_settings.get("ollama_num_predict", 0) if user_settings else 0
    user_num_ctx     = user_settings.get("ollama_num_ctx",     0) if user_settings else 0
    if user_num_predict > 0:
        base_num_predict = user_num_predict
    if user_num_ctx > 0:
        base_num_ctx = user_num_ctx

    num_predict = base_num_predict
    num_ctx     = base_num_ctx

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

    # ═══════════════════════════════════════════════════════════════════════════
    # NOVO PROMPT — Segmentação fina ao nível do discurso
    # ═══════════════════════════════════════════════════════════════════════════
    prompt = f'''{context_block}/no_think
Analisa este trecho em PT-PT. Divide-o em SEGMENTOS INDIVIDUAIS de acordo com a voz que está a falar.

REGRAS DE SEGMENTAÇÃO:

1. **Narração**: texto descritivo, ações, cenários, pensamentos do narrador (sem aspas/travessão) → `type: "narration"`
2. **Diálogo**: fala direta (entre aspas " ", « », ou travessão —) → `type: "dialogue"`
3. **Tag de diálogo**: palavras como "disse ele", "respondeu Maria", "gritou", "perguntou ela" → `type: "dialogue_tag"`
4. **Pensamento**: pensamentos internos (geralmente sem aspas, mas com verbos como "pensou", "refletiu") → `type: "thought"`
5. **Carta/nota**: texto com formatação especial ou citação longa → `type: "letter"`

CRITÉRIOS DE QUEBRA:
- **SEMPRE** que muda o falante, cria um novo segmento.
- **SEMPRE** que a emoção ou o tipo muda, cria um novo segmento.
- **NUNCA** combines narração com diálogo no mesmo segmento.
- **NUNCA** combines um diálogo com a sua tag no mesmo segmento.
- **NUNCA** combines duas falas de personagens diferentes no mesmo segmento.

REGRAS ESPECIAIS PARA NARRATIVA EM PRIMEIRA PESSOA:
- Se o capítulo/trecho é narrado por uma personagem (ex: "LÍVIA", "PAULO"), TODOS os pensamentos, ações e sentimentos dessa personagem são da responsabilidade dela, NÃO do narrador omnisciente.
- Ou seja, se o texto diz "Meu coração acelerou" e o capítulo é da Lívia, o speaker é "livia", não "narrator".
- O "narrator" só deve ser usado para:
  1. Texto descritivo em terceira pessoa.
  2. Tags de diálogo que não pertencem a ninguém em específico.
  3. Textos introdutórios (capa, ficha técnica).

ATRIBUIÇÃO DE DIÁLOGOS:
- Se uma fala é precedida por travessão (—) ou aspas, o falante é a personagem que está a falar, NUNCA a personagem que está a narrar o capítulo, a menos que o contexto indique que é ela própria.
- Exemplo: no capítulo da Lívia, se Luana diz "— Liv, vem cá!", o falante é LUANA, não Lívia.
- Para identificar o falante, procura por verbos dicendi (disse, perguntou, gritou) e nomes próprios no mesmo parágrafo.
- Se não houver pista, usa a personagem que apareceu mencionada imediatamente antes.

DISCURSO DIRETO vs. DISCURSO INDIRETO (MUITO IMPORTANTE — erro comum a evitar):
- DIRETO: a fala é reproduzida literalmente, entre aspas/travessão. O `speaker` é a personagem que fala. → `type: "dialogue"`.
- INDIRETO: o narrador RELATA o que foi dito ou pensado, sem aspas/travessão, tipicamente com um verbo dicendi (disse, respondeu, perguntou, pensou, sugeriu, admitiu, concordou, negou) seguido de "que" ou "se".
  Exemplo: "Maria disse que viria mais tarde." ou "Ele perguntou se ela tinha visto o correio."
  - Isto é SEMPRE `type: "narration"` e o `speaker` é "narrator" — NUNCA a personagem mencionada (Maria/Ele), mesmo que o nome dela apareça na própria frase. A personagem não está a falar no segmento; é o narrador que descreve o que ela disse.
  - Exceção: se este trecho está dentro de um capítulo narrado em 1ª pessoa por essa mesma personagem (ver REGRAS ESPECIAIS acima), o speaker é ela própria, não "narrator".
- INDIRETO LIVRE: pensamento relatado sem verbo declarativo explícito e sem aspas (ex: "Não podia ser verdade. Ele tinha a certeza absoluta."). → `type: "thought"`, com `speaker` = a personagem cujo pensamento é esse, se for claro pelo contexto; caso contrário "narrator".
- Regra prática para decidir "narrator" vs. personagem num trecho sem aspas/travessão: só atribui a uma personagem específica se for pensamento/sentimento dela (thought) ou narração em 1ª pessoa dela; relato de FALA de terceiros (verbo dicendi + que/se) é sempre "narrator".

EXEMPLO:
Entrada: "João entrou na sala. Estava cansado. — Finalmente chegaste! — disse Maria. Ela contou-lhe que tinha estado à espera desde cedo."
Saída:
[
  {{"speaker": "narrador", "type": "narration", "emotion": "neutral", "text": "João entrou na sala. Estava cansado."}},
  {{"speaker": "maria", "type": "dialogue", "emotion": "joyful", "text": "Finalmente chegaste!"}},
  {{"speaker": "narrador", "type": "dialogue_tag", "emotion": "neutral", "text": "disse Maria."}},
  {{"speaker": "narrador", "type": "narration", "emotion": "neutral", "text": "Ela contou-lhe que tinha estado à espera desde cedo."}}
]

PERSONAGENS JÁ CONHECIDAS (usa estes IDs sempre que possível):
{known_list}

{alias_text}

TEXTO A ANALISAR:
""" {text} """

Responde APENAS com JSON válido, com a estrutura:
{{"segments": [{{"speaker": "id", "type": "tipo", "emotion": "emoção", "text": "..."}}, ...]}}

Emoções permitidas: neutral, calm, tense, joyful, sad, angry, fearful, whisper.
'''

    # ─── SCHEMA JSON ──────────────────────────────────────────────────────────
    json_schema = {
        "type": "object",
        "properties": {
            "segments": {"type": "array"}
        },
        "required": ["segments"]
    }

    # ─── LOOP DE TENTATIVAS ──────────────────────────────────────────────────
    had_retry = False
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

            # Limpeza
            raw = raw.replace('```json', '').replace('```', '').strip()
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

            is_truncated = False
            if open_braces > close_braces or open_brackets > close_brackets:
                is_truncated = True
                had_retry = True
                logger.warning(
                    f"JSON truncado: {open_braces}x'{{' vs {close_braces}x'}}', "
                    f"{open_brackets}x'[' vs {close_brackets}x']' — "
                    f"tamanho={len(raw)} chars"
                )

            if is_truncated:
                raw = repair_truncated_json(raw)
                logger.debug(f"JSON reparado: {raw[:200]}...")

            try:
                result = json.loads(raw)

                # ── NORMALIZAR A RESPOSTA ──────────────────────────────────────
                # A resposta deve ter "segments" como lista de objetos
                if "segments" not in result or not isinstance(result["segments"], list):
                    logger.warning(f"Resposta sem 'segments' válido. A devolver vazio.")
                    return {"segments": [], "_had_retry": had_retry}

                normalized_segments = []
                for seg in result["segments"]:
                    if not isinstance(seg, dict):
                        continue
                    # Extrair campos
                    speaker = seg.get("speaker", "narrator")
                    if not speaker:
                        speaker = "narrator"
                    speaker = str(speaker).strip().lower().replace(" ", "_")

                    seg_type = seg.get("type", "narration")
                    if seg_type not in ("narration", "dialogue", "dialogue_tag", "thought", "letter"):
                        seg_type = "narration"

                    emotion = map_emotion(seg.get("emotion", "neutral"))

                    text = seg.get("text", "").strip()
                    if not text:
                        continue

                    normalized_segments.append({
                        "text": text,
                        "character_id": speaker,
                        "emotion": emotion,
                        "type": seg_type,
                        "pace": 1.0,
                        "pause_ms": 0  # será calculado depois
                    })

                result["segments"] = normalized_segments
                result["_had_retry"] = had_retry
                return result

            except json.JSONDecodeError as e:
                logger.warning(f"JSON inválido (tentativa {attempt+1}/{max_retries+1}): {e}")
                had_retry = True
                if attempt < max_retries:
                    num_predict = min(num_predict * 2, 131072)
                    num_ctx = min(num_ctx * 2, 65536)
                    logger.warning(f"A aumentar num_predict para {num_predict} e num_ctx para {num_ctx} e a tentar novamente")
                    continue
                return {"segments": [], "_had_retry": True}

        except requests.exceptions.Timeout:
            had_retry = True
            if attempt < max_retries:
                logger.warning(f"Timeout ({text_length} chars), tentativa {attempt + 1}/{max_retries + 1}")
                num_ctx = min(num_ctx * 2, 65536)
                num_predict = min(num_predict * 2, num_ctx - 2048)
                if num_ctx <= 8192:
                    timeout = small
                elif num_ctx <= 16384:
                    timeout = medium
                else:
                    timeout = large
                logger.warning(f"A tentar novamente: num_ctx={num_ctx}, num_predict={num_predict}, timeout={timeout}s")
                continue
            return {"segments": [], "_had_retry": True}

        except Exception as e:
            logger.warning(f"Erro Ollama: {e}")
            had_retry = True
            if attempt < max_retries:
                continue
            return {"segments": [], "_had_retry": True}

    return {"segments": [], "_had_retry": True}


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
4. Inclui também termos como "o pai", "a mãe", "o senhor", "a senhora", se aparecerem.
5. Para personagens sem nome próprio (ex: id "vagabundo" ou "jovem_empregado"), inclui termos como "o vagabundo", "um vagabundo", "o homem", etc., se aparecerem no texto.
6. Responde APENAS com JSON válido no formato (os IDs abaixo são apenas exemplos de estrutura — usa SEMPRE os IDs reais da lista de PERSONAGENS acima):
   {{"<character_id_1>": ["nome_proprio", "alcunha"], "<character_id_2>": ["pai", "o pai"], "<character_id_3>": ["o vagabundo", "homem"]}}

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