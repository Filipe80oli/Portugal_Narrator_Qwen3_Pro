# tts/vram_manager.py
# ─── Gestão de VRAM: Ollama unload + carregamento seletivo de modelos TTS ─────

import gc
import logging
import requests

logger = logging.getLogger(__name__)


# ─── Ollama ───────────────────────────────────────────────────────────────────

def unload_ollama(ollama_base_url: str, model_name: str, log_fn=None) -> bool:
    """
    Força o Ollama a libertar a VRAM do modelo carregado.
    Usa keep_alive=0 para descarregar imediatamente.
    Retorna True se bem-sucedido.
    """
    log = log_fn or logger.info
    try:
        log(f"🧹 A libertar VRAM do Ollama ({model_name})...")
        r = requests.post(
            f"{ollama_base_url}/api/generate",
            json={"model": model_name, "keep_alive": 0},
            timeout=30,
        )
        if r.status_code in (200, 204):
            log("✅ Ollama VRAM libertada.")
            return True
        else:
            log(f"⚠️ Ollama respondeu {r.status_code} ao unload.")
            return False
    except Exception as e:
        log(f"⚠️ Não foi possível descarregar Ollama: {e}")
        return False


# ─── TTS ─────────────────────────────────────────────────────────────────────

def release_model(model_attr: str, engine, log_fn=None) -> None:
    """
    Liberta um modelo TTS da VRAM e força o garbage collector.
    `model_attr`: "model_base" ou "model_design"
    """
    import torch
    log = log_fn or logger.info
    model = getattr(engine, model_attr, None)
    if model is None:
        return
    log(f"🧹 A libertar modelo TTS '{model_attr}' da VRAM...")
    try:
        # Mover para CPU antes de apagar (evita fragmentação)
        if hasattr(model, "to"):
            model.to("cpu")
    except Exception:
        pass
    setattr(engine, model_attr, None)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    log(f"✅ '{model_attr}' libertado.")


def vram_free_mb() -> int:
    """Retorna VRAM livre em MB (0 se não houver GPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return free // (1024 * 1024)
    except Exception:
        pass
    return 0


def log_vram(log_fn=None) -> None:
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            used = (total - free) // (1024 * 1024)
            tot  = total // (1024 * 1024)
            log  = log_fn or logger.info
            log(f"   💾 VRAM: {used} MB usados / {tot} MB total  "
                f"({free//(1024*1024)} MB livres)")
    except Exception:
        pass
