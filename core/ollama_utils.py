# core/ollama_utils.py
import asyncio
import logging
import time
import subprocess
import os
import requests
try:
    import aiohttp
except ImportError:
    aiohttp = None

logger = logging.getLogger(__name__)

# CORREÇÃO CRÍTICA: Usar 1 ano em segundos (31536000) em vez de -1.
# O Ollama (Go) tem bugs com o inteiro -1 em algumas versões, dando erro de "duration".
# 31536000 segundos = 1 ano. Garante que o modelo nunca é descarregado.
_KEEP_ALIVE = 31536000 

def ensure_ollama_server(base_url: str) -> bool:
    """
    Verifica se o servidor Ollama está a correr. 
    Se não estiver, tenta arrancá-lo automaticamente em background.
    """
    try:
        # 1. Tenta comunicar com a API (timeout curto)
        requests.get(f"{base_url}/api/tags", timeout=2)
        logger.info("✅ Servidor Ollama já está em execução.")
        return True
    except requests.ConnectionError:
        logger.warning("⚠️ Servidor Ollama não detetado. A tentar arrancar automaticamente...")
    
    # 2. Tenta arrancar o servidor em background
    try:
        cmd = "ollama"
        # Flags para não abrir janela de consola no Windows
        creation_flags = 0
        if os.name == 'nt':
            creation_flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            
        subprocess.Popen(
            [cmd, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags
        )
        
        # 3. Aguardar o servidor ficar pronto (até 15 segundos)
        for i in range(15):
            time.sleep(1)
            try:
                requests.get(f"{base_url}/api/tags", timeout=2)
                logger.info("✅ Servidor Ollama arrancado com sucesso pela App!")
                return True
            except requests.ConnectionError:
                continue
                
        logger.error("❌ O servidor Ollama não respondeu após 15s. Verifica a instalação.")
        return False
        
    except FileNotFoundError:
        logger.error("❌ Comando 'ollama' não encontrado no PATH. Instala o Ollama no sistema.")
        return False
    except Exception as e:
        # Se der erro (ex: porta já ocupada por outro processo), pode ser que já esteja a correr
        logger.debug(f"Nota ao arrancar servidor: {e}")
        try:
            requests.get(f"{base_url}/api/tags", timeout=2)
            return True # Se conseguiu conectar agora, está tudo bem
        except:
            return False

def _ensure_aiohttp():
    if aiohttp is None:
        raise ImportError(
            "O pacote 'aiohttp' é necessário para esta funcionalidade. "
            "Instale com: pip install aiohttp"
        )

# ═══════════════════════════════════════════════════════════════════════════════
# FUNÇÃO PRINCIPAL: AGUARDAR MODELO
# ═══════════════════════════════════════════════════════════════════════════════

async def wait_for_model_async(
    base_url: str,
    model_name: str,
    max_wait: int = 300,
    check_interval: int = 3,
    progress_callback=None
) -> bool:
    """
    Aguarda até que o modelo Ollama esteja carregado, pronto e estável em memória.
    Se não estiver carregado, envia um prompt de aquecimento (warmup) para forçar o carregamento.
    """
    _ensure_aiohttp()
    start_time = time.time()
    last_status = None

    if progress_callback:
        progress_callback(0.05, f"A verificar/carregar {model_name}...")
    logger.info(f"⏳ A verificar / carregar modelo {model_name}...")

    # 1. Iniciar o Warmup IMEDIATAMENTE em background
    warmup_task = asyncio.create_task(
        _send_warmup_request(base_url, model_name, "Aquecimento do modelo.")
    )

    # 2. Verificação rápida se o modelo já está carregado e estável
    status = await _get_model_status(base_url, model_name)
    if status == "loaded":
        if await _test_model_ready(base_url, model_name):
            await asyncio.sleep(2)
            if await _get_model_status(base_url, model_name) == "loaded":
                if not warmup_task.done():
                    warmup_task.cancel()
                logger.info(f"✅ Modelo {model_name} já estava carregado e estável.")
                if progress_callback:
                    progress_callback(0.3, f"{model_name} pronto!")
                return True
        else:
            logger.warning("⚠️ Modelo aparece no /api/ps mas não responde. A aguardar warmup...")

    # 3. Loop de monitorização até o modelo carregar e responder
    while (time.time() - start_time) < max_wait:
        status = await _get_model_status(base_url, model_name)
        
        if status == "loaded":
            if await _test_model_ready(base_url, model_name):
                await asyncio.sleep(2)
                if await _get_model_status(base_url, model_name) == "loaded":
                    if not warmup_task.done():
                        warmup_task.cancel()
                    logger.info(f"✅ Modelo {model_name} carregado com sucesso e estável.")
                    if progress_callback:
                        progress_callback(0.3, f"{model_name} pronto!")
                    return True
            else:
                logger.info("   Modelo carregado mas ainda não responde (a processar...)")
        else:
            if status != last_status:
                logger.info(f"   Estado do modelo: {status}")
                last_status = status

        elapsed = time.time() - start_time
        progress = min(0.3, 0.05 + 0.25 * (elapsed / max_wait))
        if progress_callback:
            progress_callback(progress, f"A carregar {model_name}... ({int(elapsed)}s)")

        await asyncio.sleep(check_interval)

    logger.warning(f"⏰ Timeout: {model_name} não ficou estável em {max_wait}s. Continuando mesmo assim.")
    return False

# ═══════════════════════════════════════════════════════════════════════════════
# FUNÇÕES AUXILIARES (INTERNAS)
# ═══════════════════════════════════════════════════════════════════════════════

async def _get_model_status(base_url: str, model_name: str) -> str:
    """
    Consulta /api/ps para saber se o modelo está carregado.
    Retorna: 'loaded', 'unloaded' ou 'error'.
    """
    _ensure_aiohttp()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/api/ps", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return "error"
                data = await resp.json()
                models = data.get("models", [])
                for m in models:
                    if m.get("name") == model_name or m.get("name", "").startswith(model_name):
                        return "loaded"
                return "unloaded"
    except Exception as e:
        logger.debug(f"Erro ao consultar /api/ps: {e}")
        return "error"

async def _test_model_ready(base_url: str, model_name: str) -> bool:
    """
    Envia um pedido de geração mínimo para verificar se o modelo responde.
    - keep_alive=-1 (inteiro) impede o Ollama de descarregar o modelo.
    - Usa /api/generate para evitar erro 400.
    """
    _ensure_aiohttp()
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": model_name,
                "prompt": "Teste",
                "stream": False,
                "keep_alive": _KEEP_ALIVE,  # INTEIRO -1 (Corrigido)
                "options": {
                    "num_predict": 1,
                    "num_ctx": 2048
                }
            }
            timeout = aiohttp.ClientTimeout(total=60)
            
            async with session.post(
                f"{base_url}/api/generate",
                json=payload,
                timeout=timeout
            ) as resp:
                if resp.status == 200:
                    return True
                
                text = await resp.text()
                logger.debug(f"Teste de prontidão falhou: {resp.status} - {text[:200]}")
                return False
                
    except asyncio.TimeoutError:
        logger.debug("Timeout no teste de prontidão.")
        return False
    except Exception as e:
        logger.debug(f"Erro de exceção no teste de prontidão: {e}")
        return False

async def _send_warmup_request(base_url: str, model_name: str, prompt: str):
    """
    Envia uma requisição de geração para forçar o carregamento do modelo.
    - keep_alive=-1 (inteiro) mantém o modelo em memória.
    """
    _ensure_aiohttp()
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "keep_alive": _KEEP_ALIVE,  # INTEIRO -1 (Corrigido)
                "options": {
                    "num_predict": 1,
                    "num_ctx": 2048
                }
            }
            timeout = aiohttp.ClientTimeout(total=600)
            
            async with session.post(
                f"{base_url}/api/generate",
                json=payload,
                timeout=timeout
            ) as resp:
                if resp.status == 200:
                    logger.debug("✅ Warmup request concluída com sucesso.")
                else:
                    text = await resp.text()
                    logger.warning(f"⚠️ Warmup falhou: {resp.status} - {text[:200]}")
                    
    except asyncio.CancelledError:
        logger.debug("Warmup cancelado (modelo já estava pronto).")
    except asyncio.TimeoutError:
        logger.warning("⏰ Timeout no warmup (demorou mais de 10 minutos).")
    except Exception as e:
        logger.warning(f"❌ Erro inesperado no warmup: {e}")