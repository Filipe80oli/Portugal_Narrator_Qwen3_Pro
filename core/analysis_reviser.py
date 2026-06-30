# core/analysis_reviser.py
import re
import logging
import requests
from typing import Dict, List, Tuple, Set

logger = logging.getLogger(__name__)


def get_ollama_response(prompt: str, ollama_url: str, model_name: str, timeout: int = 60) -> str:
    try:
        resp = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "keep_alive": -1,   # Bug 1: sem keep_alive o modelo descarregava entre revisões
                "options": {"temperature": 0.0, "num_predict": 50}
            },
            timeout=timeout
        )
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
        return ""
    except Exception:
        return ""


def build_speaker_prompt(text: str, character_names: List[str]) -> str:
    names_list = ", ".join(character_names)
    if len(text) > 800:
        text = text[:800] + "..."
    prompt = f"""Dado o seguinte excerto de um livro, identifica qual personagem está a falar (discurso direto).
A lista de personagens é: {names_list}.
Se não houver um falante claro ou se for narração, responde apenas com a palavra "narrator".
Excerto:
---
{text}
---
Responde apenas com o nome exato da personagem (como está na lista) ou "narrator"."""
    return prompt


def find_missing_text(
    original_text: str,
    segments: List[dict],
    min_chars: int = 50,
    log_fn=None
) -> List[str]:
    """
    Compara o texto original com os segmentos da análise e identifica
    blocos de texto que foram omitidos.

    Bugs corrigidos:
    - Bug 2+3: o bloco de contagem de frases estava DENTRO do loop `for seg_text`,
      pelo que reiniciava matched_sentences a 0 para cada segmento e só comparava
      as frases com o seg_text da iteração corrente — nunca com todos os segmentos.
      Movido para FORA do loop seg_text, percorrendo segment_texts independentemente.
    - Bug 4: o limiar de 60% dividia por len(sentences) que inclui frases curtas
      ignoradas pelo `if len(sent) < 30`. Agora divide pelo número de frases longas.
    """
    if log_fn is None:
        log_fn = logger.info

    # 1. Construir conjunto normalizado de todos os textos dos segmentos
    segment_texts: Set[str] = set()
    for seg in segments:
        text = seg.get("text", "").strip()
        if text:
            segment_texts.add(" ".join(text.split()))

    # 2. Dividir o texto original em parágrafos
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', original_text) if p.strip()]

    missing_blocks = []
    for para in paragraphs:
        if len(para) < min_chars:
            continue

        normalized_para = " ".join(para.split())
        found = False

        # Verificação exacta: parágrafo contido num segmento ou vice-versa
        for seg_text in segment_texts:
            if (normalized_para in seg_text) or (seg_text in normalized_para):
                found = True
                break

        # Bug 2+3 fix: verificação por frases, fora do loop de seg_text
        if not found and len(para) > 500:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            long_sentences = [s for s in sentences if len(s) >= 30]

            if long_sentences:
                matched = 0
                for sent in long_sentences:
                    normalized_sent = " ".join(sent.split())
                    for seg_text in segment_texts:
                        if normalized_sent in seg_text or seg_text in normalized_sent:
                            matched += 1
                            break  # este sent está coberto, passar ao seguinte

                # Bug 4 fix: dividir por long_sentences, não por todas as sentences
                if matched >= len(long_sentences) * 0.6:
                    found = True

        if not found:
            preview = para[:200] + "..." if len(para) > 200 else para
            missing_blocks.append(preview)

    if missing_blocks:
        log_fn(f"   ⚠️ Detetados {len(missing_blocks)} blocos de texto que podem estar em falta:")
        for i, block in enumerate(missing_blocks[:5]):
            log_fn(f"      [{i+1}] {block}")
        if len(missing_blocks) > 5:
            log_fn(f"      ... e mais {len(missing_blocks) - 5} blocos.")
    else:
        log_fn("   ✅ Todos os parágrafos principais estão representados nos segmentos.")

    return missing_blocks


def revise_analysis(
    segments: List[dict],
    characters: Dict[str, dict],
    ollama_base_url: str,
    model_name: str,
    original_text: str = "",
    max_segments: int = 200,
    log_fn=None
) -> Tuple[List[dict], Dict[int, str], List[str]]:
    """
    Revisa a atribuição de falas e verifica se há texto em falta.
    Retorna: (segments, corrections, missing_blocks)

    Bug 5 fix: trabalhar sobre uma cópia dos segmentos para não mutar a lista original.
    """
    if log_fn is None:
        log_fn = logger.info

    # Bug 5: copiar para não mutar a lista original passada pelo chamador
    segments = [seg.copy() for seg in segments]

    # ---- Parte 1: correção de falas ----------------------------------------
    name_to_id: Dict[str, str] = {}
    for cid, cdata in characters.items():
        if not isinstance(cdata, dict):
            continue
        nome = cdata.get("name", "").strip()
        if nome:
            name_to_id[nome.lower()] = cid

    dialogue_pattern = re.compile(r'["«»]|—')
    candidates = []
    for idx, seg in enumerate(segments):
        text = seg.get("text", "")
        if not text or len(text) < 10:
            continue
        if dialogue_pattern.search(text):
            candidates.append((idx, seg))

    log_fn(f"🔍 A revisar atribuição de falas em {len(candidates)} segmentos com diálogo...")

    if len(candidates) > max_segments:
        log_fn(f"   ⚠️ Apenas os primeiros {max_segments} serão revistos.")
        candidates = candidates[:max_segments]

    corrections: Dict[int, str] = {}
    ollama_url = ollama_base_url.rstrip("/")

    for idx, seg in candidates:
        text = seg.get("text", "")
        current_id = seg.get("character_id", "narrator")
        names = [
            cdata["name"]
            for cdata in characters.values()
            if isinstance(cdata, dict) and cdata.get("name")
        ]
        prompt = build_speaker_prompt(text, names)
        resposta = get_ollama_response(prompt, ollama_url, model_name)

        if not resposta:
            continue

        resposta_clean = resposta.strip().lower()
        if resposta_clean in ("narrator", "narrador", "narradora"):
            suggested_id = "narrator"
        else:
            suggested_id = name_to_id.get(resposta_clean)
            if not suggested_id:
                for nome, cid in name_to_id.items():
                    if resposta_clean in nome or nome in resposta_clean:
                        suggested_id = cid
                        break

        if suggested_id and suggested_id != current_id:
            log_fn(f"   ✏️ Segmento {idx}: '{current_id}' → '{suggested_id}' (modelo sugeriu '{resposta}')")
            corrections[idx] = suggested_id

    for idx, new_id in corrections.items():
        segments[idx]["character_id"] = new_id

    log_fn(f"✅ Revisão de falas: {len(corrections)} correções aplicadas.")

    # ---- Parte 2: verificar texto em falta ---------------------------------
    missing_blocks: List[str] = []
    if original_text:
        log_fn("📄 A verificar possíveis omissões de texto...")
        missing_blocks = find_missing_text(original_text, segments, log_fn=log_fn)

    return segments, corrections, missing_blocks