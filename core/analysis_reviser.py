# core/analysis_reviser.py
import re
import json
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


def build_speaker_prompt_batch(items: List[Tuple[int, str]], character_names: List[str]) -> str:
    """
    Constrói um único prompt que pede ao modelo para identificar o falante
    de VÁRIOS excertos de uma só vez, em vez de um pedido por segmento.

    Otimização de desempenho: cada pedido ao Ollama tem overhead fixo
    (processamento de prompt, arranque de geração) independentemente do
    tamanho do texto. Agrupar N segmentos num único pedido reduz N
    round-trips sequenciais para 1, o que é a diferença entre, por exemplo,
    19 pedidos de ~10-15s cada (~3-5 min) e 2-3 pedidos (~30-60s no total).
    """
    names_list = ", ".join(character_names)
    excerpt_lines = []
    for local_idx, text in items:
        snippet = text if len(text) <= 500 else text[:500] + "..."
        excerpt_lines.append(f'{local_idx}. "{snippet}"')
    excerpts_block = "\n".join(excerpt_lines)

    prompt = f"""Dado o seguinte livro, identifica qual personagem está a falar em CADA excerto numerado (discurso direto).
A lista de personagens é: {names_list}.
Se um excerto não tiver um falante claro ou for narração, usa "narrator".

EXCERTOS:
{excerpts_block}

Responde APENAS com JSON válido no formato:
{{"respostas": [{{"id": <número>, "falante": "<nome_exato_ou_narrator>"}}, ...]}}
Inclui uma entrada para CADA excerto numerado acima, na mesma ordem."""
    return prompt


def get_ollama_response_batch(
    prompt: str, ollama_url: str, model_name: str, n_items: int, timeout: int = 120
) -> Dict[int, str]:
    """
    Envia o prompt em lote e devolve um dict {id_local: falante}.
    Em caso de falha (timeout, JSON inválido), devolve dict vazio —
    o chamador trata isso como "sem correções para este lote" em vez
    de rebentar, preservando o comportamento anterior de fail-soft.
    """
    json_schema = {
        "type": "object",
        "properties": {"respostas": {"type": "array"}},
        "required": ["respostas"]
    }
    try:
        resp = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "format": json_schema,
                "stream": False,
                "keep_alive": -1,
                "options": {
                    "temperature": 0.0,
                    "num_ctx": 8192,
                    "num_predict": max(512, n_items * 40),
                }
            },
            timeout=timeout
        )
        if resp.status_code != 200:
            return {}
        raw = resp.json().get("response", "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        results = {}
        for item in data.get("respostas", []):
            if not isinstance(item, dict):
                continue
            try:
                local_id = int(item.get("id"))
            except (TypeError, ValueError):
                continue
            falante = str(item.get("falante", "")).strip()
            if falante:
                results[local_id] = falante
        return results
    except Exception as e:
        logger.debug(f"Falha na revisão em lote: {e}")
        return {}


def _normalize_for_comparison(text: str) -> str:
    """
    Normaliza texto para comparação tolerante a diferenças tipográficas.

    BUG CORRIGIDO: find_missing_text comparava o texto original do livro
    (que normalmente usa aspas curvas “ ” ‘ ’) com o texto dos segmentos
    devolvidos pelo Ollama (que frequentemente usa aspas retas " ' ou tem
    espaçamento ligeiramente diferente à volta da pontuação). Isto fazia
    com que diálogos genuinamente presentes nos segmentos fossem reportados
    como "texto em falta" — falsos positivos. Esta normalização converte
    aspas tipográficas para retas e colapsa espaços antes de comparar,
    tal como já é feito em core.text_normalizer.normalize_text.
    """
    if not text:
        return ""
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = " ".join(text.split())
    return text


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
    - Bug 5: comparação não tolerava aspas curvas vs retas — ver _normalize_for_comparison.
    """
    if log_fn is None:
        log_fn = logger.info

    # 1. Construir conjunto normalizado de todos os textos dos segmentos
    segment_texts: Set[str] = set()
    for seg in segments:
        text = seg.get("text", "").strip()
        if text:
            segment_texts.add(_normalize_for_comparison(text))

    # 2. Dividir o texto original em parágrafos
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', original_text) if p.strip()]

    missing_blocks = []
    for para in paragraphs:
        if len(para) < min_chars:
            continue

        normalized_para = _normalize_for_comparison(para)
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
                    normalized_sent = _normalize_for_comparison(sent)
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
    names = [
        cdata["name"]
        for cdata in characters.values()
        if isinstance(cdata, dict) and cdata.get("name")
    ]

    # ── REVISÃO EM LOTE ──────────────────────────────────────────────────
    # Em vez de 1 pedido ao Ollama por segmento (N round-trips sequenciais,
    # cada um com overhead fixo de processamento de prompt), agrupamos
    # BATCH_SIZE segmentos por pedido. Para 19 segmentos isto passa de
    # 19 pedidos para ~3, reduzindo o tempo total de vários minutos para
    # dezenas de segundos.
    BATCH_SIZE = 8
    batches = [candidates[i:i + BATCH_SIZE] for i in range(0, len(candidates), BATCH_SIZE)]

    for batch_num, batch in enumerate(batches):
        items = [(local_i, seg.get("text", "")) for local_i, (idx, seg) in enumerate(batch)]
        prompt = build_speaker_prompt_batch(items, names)
        batch_results = get_ollama_response_batch(
            prompt, ollama_url, model_name, n_items=len(batch)
        )

        # Fallback: se o lote falhou (dict vazio) ou veio incompleto,
        # processar individualmente apenas os que faltaram — preserva
        # a robustez do comportamento anterior sem perder a vantagem de
        # velocidade nos casos (normais) em que o lote funciona.
        missing_local_ids = [i for i in range(len(batch)) if i not in batch_results]
        for local_i in missing_local_ids:
            idx, seg = batch[local_i]
            text = seg.get("text", "")
            prompt_single = build_speaker_prompt(text, names)
            resposta = get_ollama_response(prompt_single, ollama_url, model_name)
            if resposta:
                batch_results[local_i] = resposta

        for local_i, (idx, seg) in enumerate(batch):
            resposta = batch_results.get(local_i)
            if not resposta:
                continue
            current_id = seg.get("character_id", "narrator")
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

        log_fn(f"   📦 Lote {batch_num + 1}/{len(batches)} processado ({len(batch)} segmentos).")

    for idx, new_id in corrections.items():
        segments[idx]["character_id"] = new_id

    log_fn(f"✅ Revisão de falas: {len(corrections)} correções aplicadas.")

    # ---- Parte 2: verificar texto em falta ---------------------------------
    missing_blocks: List[str] = []
    if original_text:
        log_fn("📄 A verificar possíveis omissões de texto...")
        missing_blocks = find_missing_text(original_text, segments, log_fn=log_fn)

    return segments, corrections, missing_blocks