# cinema/sound_analyzer.py
# ─── Deteção de eventos sonoros nos segmentos (via Ollama) ────────────────────

import re
import json
import logging
import requests

logger = logging.getLogger(__name__)


SOUND_PROMPT_TEMPLATE = """Analisa este trecho de texto narrativo e identifica TODOS os eventos sonoros implícitos ou explícitos.

TEXTO:
\"\"\"{text}\"\"\"

Para cada evento sonoro indica:
- "sound": nome curto do som em inglês (ex: "rain", "door_knock", "thunder", "footsteps", "crowd", "fire_crackle")
- "duration": duração estimada em segundos (0.5 a 10.0) conforme a intensidade narrativa
- "volume": -20 a 0 dB (0 = forte, -20 = suave/fundo)
- "position": "before" (antes da fala), "during" (em paralelo) ou "after" (depois)
- "description_pt": breve descrição em português do que se ouve

Inclui também música de fundo se o tom emocional justificar:
- "music": estilo em inglês (ex: "dramatic", "tense", "romantic", "sad", "happy", "mystery", "peaceful")
- "music_volume": -30 a -10 dB
- "music_duration": duração em segundos (5 a 30)

Se não houver sons relevantes, retorna listas vazias.

Responde APENAS com JSON neste formato:
{{
  "sounds": [
    {{"sound": "rain_heavy", "duration": 3.0, "volume": -8, "position": "during", "description_pt": "Chuva intensa de fundo"}},
    {{"sound": "thunder", "duration": 1.5, "volume": -3, "position": "before", "description_pt": "Trovão repentino"}}
  ],
  "music": {{"music": "tense", "music_volume": -20, "music_duration": 15.0}}
}}"""


async def analyze_sounds_for_segment(
    ollama_url: str, model_name: str, text: str, emotion: str
) -> dict:
    """
    Chama o Ollama para detetar sons e música no segmento.
    Retorna dict com 'sounds' (list) e 'music' (dict|None).
    """
    prompt = SOUND_PROMPT_TEMPLATE.format(text=text[:1500])

    json_schema = {
        "type": "object",
        "properties": {
            "sounds": {"type": "array"},
            "music": {"type": "object"}
        },
        "required": ["sounds"]
    }

    try:
        r = requests.post(ollama_url, json={
            "model": model_name,
            "prompt": prompt,
            "format": json_schema,
            "stream": False,
            "options": {"temperature": 0, "top_k": 1}
        }, timeout=120)
        r.raise_for_status()
        raw = r.json().get("response", "").strip()
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            matches = list(re.finditer(r'\{.*\}', raw, flags=re.DOTALL))
            data = json.loads(matches[-1].group()) if matches else {}

        return {
            "sounds": _validate_sounds(data.get("sounds", [])),
            "music": _validate_music(data.get("music")),
        }

    except Exception as e:
        logger.warning(f"Erro ao analisar sons: {e}")
        return {"sounds": [], "music": None}


async def analyze_sounds_batch(
    ollama_url: str, model_name: str,
    segments: list, progress_fn=None
) -> list[dict]:
    """
    Analisa sons para todos os segmentos narrativos (narrator + emotions).
    Retorna lista paralela a `segments` com os eventos sonoros.
    """
    import asyncio
    results = []
    # Apenas segmentos com texto substantivo valem a análise
    for i, seg in enumerate(segments):
        text    = seg.get("text", "")
        emotion = seg.get("emotion", "neutral")

        # Análise apenas em segmentos longos (>80 chars) ou com emoção forte
        worth = (len(text) > 80 or emotion in
                 ("tense", "angry", "fearful", "sad", "joyful"))

        if worth:
            result = await analyze_sounds_for_segment(
                ollama_url, model_name, text, emotion
            )
        else:
            result = {"sounds": [], "music": None}

        results.append(result)

        if progress_fn:
            progress_fn(i / len(segments),
                        f"A analisar sons: segmento {i+1}/{len(segments)}")

        # Pequena pausa para não saturar o Ollama
        await asyncio.sleep(0.05)

    return results


# ─── Validação ────────────────────────────────────────────────────────────────

def _validate_sounds(raw: list) -> list:
    result = []
    for s in raw:
        if not isinstance(s, dict):
            continue
        result.append({
            "sound":          str(s.get("sound", "")).strip(),
            "duration":       float(s.get("duration", 2.0)),
            "volume":         int(s.get("volume", -10)),
            "position":       str(s.get("position", "during")),
            "description_pt": str(s.get("description_pt", "")),
        })
    return [s for s in result if s["sound"]]


def _validate_music(raw) -> dict | None:
    if not isinstance(raw, dict) or not raw.get("music"):
        return None
    return {
        "music":          str(raw.get("music", "dramatic")),
        "music_volume":   int(raw.get("music_volume", -20)),
        "music_duration": float(raw.get("music_duration", 10.0)),
    }
