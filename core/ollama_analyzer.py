# core/ollama_analyzer.py
# ─── Análise do livro via Ollama (deteção de personagens e segmentação) ────────

import re
import json
import logging
import requests

logger = logging.getLogger(__name__)


def get_ollama_models(base_url: str) -> list[str]:
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception as e:
        logger.warning(f"Ollama indisponível: {e}")
        return ["qwen2.5:7b", "qwen2.5:14b"]


def warmup_ollama(ollama_url: str, model_name: str):
    try:
        requests.post(ollama_url, json={
            "model": model_name, "prompt": "ok",
            "stream": False, "options": {"temperature": 0}
        }, timeout=300)
    except Exception as e:
        logger.warning(f"Warmup falhou: {e}")


def split_into_blocks(text: str, max_chars: int = 4000) -> list[str]:
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    blocks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) > max_chars and current:
            blocks.append(current)
            current = p
        else:
            current += "\n\n" + p if current else p
    if current:
        blocks.append(current)
    return blocks


def sanitize_segments(raw_segments: list) -> list:
    """Garante que todos os segmentos são dicts válidos."""
    result = []
    for item in raw_segments:
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            if text:
                result.append({
                    "text": text,
                    "character_id": str(item.get("character_id", "narrator")),
                    "emotion": str(item.get("emotion", "neutral")),
                    "pace": float(item.get("pace", 1.0)),
                    "pause_ms": int(item.get("pause_ms", 0)),
                })
        elif isinstance(item, str):
            text = item.strip()
            if text:
                result.append({
                    "text": text, "character_id": "narrator",
                    "emotion": "neutral", "pace": 1.0, "pause_ms": 0
                })
        elif isinstance(item, list):
            result.extend(sanitize_segments(item))
    return result


async def analyze_block(ollama_url: str, model_name: str,
                         text: str, context: str, known_chars: dict) -> dict | None:
    """Envia um bloco ao Ollama e retorna personagens + segmentos."""
    known_list = "\n".join([
        f"- {cid}: {c['name']} ({c['type']})"
        for cid, c in known_chars.items()
    ]) if known_chars else "(nenhum ainda)"

    context_block = f"CONTEXTO PRÉVIO (resumo):\n{context[:800]}...\n\n" if context else ""

    prompt = f"""{context_block}Analisa este trecho de um livro português (PT-PT) e faz DUAS coisas:

1. IDENTIFICA TODAS AS PERSONAGENS que falam ou são mencionadas. Para cada uma:
   - Atribui um ID único (ex: "narrator", "maria", "joao", "child_1")
   - Dá o nome (ou "Narrador" se for narração)
   - Classifica o TIPO: narrator, man, woman, young_man, young_woman, boy, girl, old_man, old_woman
   - Cria uma DESCRIÇÃO VOCAL detalhada em português para síntese de voz

2. SEGMENTA o texto em partes, atribuindo cada parte a uma personagem. Para cada segmento indica:
   - character_id (referência ao ID acima)
   - emotion: neutral, calm, tense, joyful, sad, angry, fearful, whisper
   - pace: 0.8 a 1.2 (ritmo)
   - pause_ms: 0 a 1500 (pausa antes deste segmento)

PERSONAGENS JÁ CONHECIDAS (mantém os IDs):
{known_list}

REGRAS:
- Mantém o texto 100% intacto, não inventes nem resumas.
- Se for narração (sem fala direta), usa character_id "narrator".
- Fala direta entre aspas « » ou "", ou travessão —, identifica quem fala.
- Sê consistente com os IDs entre blocos.

TEXTO PARA ANÁLISE:
\"\"\"{text}\"\"\"

Responde APENAS com JSON neste formato exato:
{{
  "characters": {{
    "narrator": {{"name": "Narrador", "type": "narrator", "description": "Voz masculina madura, tom neutro..."}},
    "maria": {{"name": "Maria", "type": "young_woman", "description": "Voz feminina jovem, tom alegre..."}}
  }},
  "segments": [
    {{"text": "...", "character_id": "narrator", "emotion": "calm", "pace": 1.0, "pause_ms": 0}},
    {{"text": "...", "character_id": "maria", "emotion": "joyful", "pace": 1.1, "pause_ms": 300}}
  ]
}}"""

    json_schema = {
        "type": "object",
        "properties": {
            "characters": {"type": "object"},
            "segments": {"type": "array"}
        },
        "required": ["characters", "segments"]
    }

    try:
        r = requests.post(ollama_url, json={
            "model": model_name, "prompt": prompt, "format": json_schema,
            "stream": False, "options": {"temperature": 0, "top_k": 1}
        }, timeout=600)
        r.raise_for_status()
        raw = r.json().get('response', '').strip()
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            matches = list(re.finditer(r'\{.*\}', raw, flags=re.DOTALL))
            return json.loads(matches[-1].group()) if matches else {}
    except Exception as e:
        logger.warning(f"Erro Ollama no bloco: {e}")
        return None
