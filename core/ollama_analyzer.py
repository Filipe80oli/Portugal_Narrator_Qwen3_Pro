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

    prompt = f"""{context_block}SEJA CONCISO E RÁPIDO. Analisa este trecho de livro PT-PT:

1. IDENTIFICA PERSONAGENS (ID único, nome, tipo, descrição vocal PT-PT)

2. SEGMENTA texto atribuindo a personagens. REGRAS:
- "..." ou «...» → personagem que fala
- resto → narrador
- cada segmento: character_id, emotion, pace, pause_ms

EXEMPLO:
Texto: "Olá!" disse Maria, sorrindo.
Segmentos:
1. {{"text": "Olá!", "character_id": "maria", "emotion": "joyful", "pace": 1.1, "pause_ms": 0}}
2. {{"text": "disse Maria, sorrindo.", "character_id": "narrator", "emotion": "neutral", "pace": 1.0, "pause_ms": 150}}

PERSONAGENS CONHECIDAS:
{known_list}

TEXTO:
\"\"\"{text}\"\"\"

RESPONDA APENAS COM JSON VÁLIDO (sem explicações):
{{
  "characters": {{
    "narrator": {{"name": "Narrador", "type": "narrator", "description": "Voz masculina madura, tom neutro, ritmo pausado e claro, sotaque português de Portugal"}}
  }},
  "segments": [
    {{"text": "...", "character_id": "narrator", "emotion": "neutral", "pace": 1.0, "pause_ms": 0}}
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
            "model": model_name, 
            "prompt": prompt, 
            "format": json_schema,
            "stream": False, 
            "options": {
                "temperature": 0, 
                "top_k": 1,
                "repeat_penalty": 1.2,
                "mirostat": 0  # Desativar modo de pensamento complexo
            }
        }, timeout=300)  # Reduzido timeout para 300s
        r.raise_for_status()
        raw = r.json().get('response', '').strip()
        
        # Limpeza mais agressiva
        raw = re.sub(r'[^{{]*({{.*)', r'\1', raw, flags=re.DOTALL).strip()
        raw = re.sub(r'(}})[^}}]*$', r'\1', raw, flags=re.DOTALL).strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Tentar extrair JSON mais agressivamente
            matches = list(re.finditer(r'\{{.*\}}', raw, flags=re.DOTALL))
            if matches:
                try:
                    return json.loads(matches[-1].group())
                except:
                    pass
            return {"characters": {}, "segments": []}
    except Exception as e:
        logger.warning(f"Erro Ollama no bloco: {e}")
        return {"characters": {}, "segments": []}


