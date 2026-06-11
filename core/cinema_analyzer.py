# core/cinema_analyzer.py
# ─── Análise cinematográfica: deteção de sons e música pelo Ollama ─────────────
#
# Para cada segmento do texto, o Ollama devolve:
#   - sfx     : lista de efeitos sonoros a misturar (nome + offset + duração)
#   - music   : música de fundo (nome + duração)
#
# Exemplo de resultado por segmento:
# {
#   "sfx": [
#     {"sound": "rain", "offset_ms": 0, "duration_s": 3.0, "volume_db": -6},
#     {"sound": "thunder", "offset_ms": 1200, "duration_s": 1.5, "volume_db": -4}
#   ],
#   "music": {"sound": "tense", "duration_s": 6.0, "volume_db": -18}
# }

import re
import json
import logging
import requests

from config.settings import (
    SOUND_CATEGORIES,
    CINEMA_SFX_DEFAULT_DURATION,
    CINEMA_MUSIC_DEFAULT_DURATION,
)

logger = logging.getLogger(__name__)

# Lista plana de todos os sons conhecidos (para o prompt)
_ALL_SOUNDS = sorted({s for sounds in SOUND_CATEGORIES.values() for s in sounds})


async def analyze_cinema_block(ollama_url: str, model_name: str,
                                segments: list[dict]) -> list[dict]:
    """
    Recebe uma lista de segmentos (já com text/character_id/emotion) e
    devolve a mesma lista enriquecida com 'sfx' e 'music' em cada segmento.
    Processa em lotes de 10 segmentos para não sobrecarregar o contexto.
    """
    enriched = []
    batch_size = 10

    for i in range(0, len(segments), batch_size):
        batch = segments[i:i + batch_size]
        result = await _analyze_batch(ollama_url, model_name, batch)
        enriched.extend(result)

    return enriched


async def _analyze_batch(ollama_url: str, model_name: str,
                          batch: list[dict]) -> list[dict]:
    """Envia um lote de segmentos ao Ollama e devolve os segmentos enriquecidos."""

    # Preparar input simplificado para o Ollama (sem texto completo, só resumo)
    batch_input = []
    for idx, seg in enumerate(batch):
        batch_input.append({
            "idx": idx,
            "text_preview": seg.get("text", "")[:120],
            "emotion": seg.get("emotion", "neutral"),
            "character": seg.get("character_id", "narrator"),
        })

    sounds_list = ", ".join(_ALL_SOUNDS)

    prompt = f"""Analisa estes {len(batch)} segmentos de um audiobook e para cada um decide:
1. Quais EFEITOS SONOROS (sfx) devem ocorrer durante este segmento (máx 2 por segmento).
2. Que MÚSICA DE FUNDO (music) deve tocar durante este segmento (pode ser null).

SONS DISPONÍVEIS NA BASE DE DADOS: {sounds_list}

REGRAS:
- Usa APENAS sons da lista acima. Se não há som adequado, usa null.
- offset_ms: milissegundos após o início do segmento de voz para iniciar o som.
- duration_s: duração em segundos do som (0.5 a 10.0).
- volume_db: -20 (muito suave) a 0 (normal). Efeitos: -8 a -4. Música: -20 a -14.
- Se o segmento for narração neutra sem ação descrita, sfx=[] e music=null.
- Sê criterioso: menos é mais. Não adiciones sons a tudo.

SEGMENTOS:
{json.dumps(batch_input, ensure_ascii=False, indent=2)}

Responde APENAS com JSON:
{{
  "results": [
    {{
      "idx": 0,
      "sfx": [
        {{"sound": "rain", "offset_ms": 0, "duration_s": 3.0, "volume_db": -6}}
      ],
      "music": {{"sound": "tense", "duration_s": 6.0, "volume_db": -18}}
    }},
    {{
      "idx": 1,
      "sfx": [],
      "music": null
    }}
  ]
}}"""

    json_schema = {
        "type": "object",
        "properties": {
            "results": {"type": "array"}
        },
        "required": ["results"]
    }

    try:
        r = requests.post(ollama_url, json={
            "model": model_name,
            "prompt": prompt,
            "format": json_schema,
            "stream": False,
            "options": {"temperature": 0, "top_k": 1}
        }, timeout=300)
        r.raise_for_status()
        raw = r.json().get('response', '').strip()
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            matches = list(re.finditer(r'\{.*\}', raw, flags=re.DOTALL))
            data = json.loads(matches[-1].group()) if matches else {"results": []}

        # Mapear resultados de volta para os segmentos
        results_map = {r["idx"]: r for r in data.get("results", []) if isinstance(r, dict)}

        enriched_batch = []
        for idx, seg in enumerate(batch):
            seg_copy = dict(seg)
            meta = results_map.get(idx, {})

            # Normalizar SFX
            sfx_raw = meta.get("sfx", [])
            seg_copy["sfx"] = _normalize_sfx(sfx_raw) if isinstance(sfx_raw, list) else []

            # Normalizar música
            music_raw = meta.get("music")
            seg_copy["music"] = _normalize_music(music_raw)

            enriched_batch.append(seg_copy)

        return enriched_batch

    except Exception as e:
        logger.warning(f"Erro análise cinema: {e}")
        # Retorna segmentos sem enriquecimento em caso de erro
        return [{**seg, "sfx": [], "music": None} for seg in batch]


def _normalize_sfx(sfx_list: list) -> list:
    result = []
    for item in sfx_list:
        if not isinstance(item, dict):
            continue
        sound = str(item.get("sound", "")).strip()
        if not sound:
            continue
        result.append({
            "sound": sound,
            "offset_ms": max(0, int(item.get("offset_ms", 0))),
            "duration_s": max(0.5, min(10.0, float(item.get("duration_s", CINEMA_SFX_DEFAULT_DURATION)))),
            "volume_db": max(-20, min(0, float(item.get("volume_db", -6)))),
        })
    return result[:2]  # máximo 2 por segmento


def _normalize_music(music_raw) -> dict | None:
    if not music_raw or not isinstance(music_raw, dict):
        return None
    sound = str(music_raw.get("sound", "")).strip()
    if not sound:
        return None
    return {
        "sound": sound,
        "duration_s": max(1.0, min(30.0, float(music_raw.get("duration_s", CINEMA_MUSIC_DEFAULT_DURATION)))),
        "volume_db": max(-30, min(-8, float(music_raw.get("volume_db", -18)))),
    }
