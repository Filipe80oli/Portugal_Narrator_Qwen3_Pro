# config/settings.py
# ─── Configurações globais da aplicação ───────────────────────────────────────

from pathlib import Path

# Modelos Qwen3-TTS
QWEN3_MODEL_BASE        = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
QWEN3_MODEL_VOICEDESIGN = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

# Diretórios
ANALYSIS_DIR = Path("analyses")
ANALYSIS_DIR.mkdir(exist_ok=True)

TEMP_DIR = Path("temp_audio_chunks")
TEMP_DIR.mkdir(exist_ok=True)

# Base de dados de sons (modo Cinema)
# Estrutura recomendada:
#   sounds/nature/rain_heavy.wav
#   sounds/nature/thunder.wav
#   sounds/city/traffic.wav
#   sounds/music/dramatic.wav   ← música de fundo
SOUNDS_DIR = Path("sounds")
SOUNDS_DIR.mkdir(exist_ok=True)

# Ollama
OLLAMA_BASE_URL      = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"

# Narrador padrão
DEFAULT_NARRATOR = {
    "name": "Narrador",
    "type": "narrator",
    "description": (
        "Voz masculina madura, tom neutro e envolvente, "
        "ritmo pausado e claro, sotaque português de Portugal"
    )
}

# Anchor PT-PT para narrador
NARRATOR_PT_PT_INSTRUCT = (
    "Voz masculina de um homem português. Sotaque de Lisboa, Portugal. "
    "Português europeu estrito. Sem qualquer cadência brasileira. "
    "Voz profunda, madura e clara de Portugal."
)

ANCHOR_TEXT = (
    "Estou a falar com o sotaque de Lisboa, em Portugal. "
    "Esta é a minha voz europeia."
)

# ─── Modos de Produção ────────────────────────────────────────────────────────
# "narrator" : uma só voz para todo o texto
# "novela"   : múltiplas vozes por personagem
# "cinema"   : novela + música ambiente + efeitos sonoros detetados pelo Ollama
PRODUCTION_MODES    = ["🎙️ Narrador", "🎭 Novela", "🎬 Cinema"]
PRODUCTION_MODE_IDS = ["narrator",    "novela",    "cinema"]

# ─── Cinema: volumes de mixagem ───────────────────────────────────────────────
CINEMA_VOICE_VOL             =   0    # dB (sem alteração)
CINEMA_SFX_VOL               =  -6    # efeitos 6 dB abaixo da voz
CINEMA_MUSIC_VOL             = -18    # música de fundo bem abaixo

CINEMA_SFX_DEFAULT_DURATION  = 2.0   # segundos
CINEMA_MUSIC_DEFAULT_DURATION = 8.0

# ─── Categorias de sons (para fallback na DB) ─────────────────────────────────
SOUND_CATEGORIES = {
    "nature":   ["rain", "thunder", "wind", "sea", "river", "fire", "birds",
                 "forest", "storm", "snow", "leaves"],
    "city":     ["traffic", "crowd", "sirens", "construction", "market",
                 "cafe", "subway", "clock", "bell"],
    "interior": ["door_open", "door_close", "door_knock", "footsteps",
                 "phone_ring", "glass_break", "typing", "clock_ticking",
                 "fire_crackle", "chair_creak"],
    "action":   ["explosion", "gunshot", "fight", "horse", "sword",
                 "car_crash", "scream", "run"],
    "ambience": ["silence", "night", "morning", "tension", "mystery",
                 "romantic", "sad", "happy", "dramatic"],
    "music":    ["dramatic", "romantic", "tense", "sad", "happy",
                 "mystery", "action", "peaceful", "epic"],
}

# ─── Qualidade de Áudio & Retry ───────────────────────────────────────────────
TTS_MAX_RETRIES         = 4      # Aumentar para mais tentativas
TTS_RETRY_TEMP_STEP     = 0.05   # Incremento menor de temperature
TTS_MIN_DURATION_RATIO  = 0.05   # Mais permissivo com duração
TTS_MAX_SILENCE_RATIO   = 0.90   # Mais permissivo com silêncio
TTS_MIN_RMS             = 0.0005 # Mais sensível a ruído baixo
TTS_MAX_RMS             = 0.95   # Menos restritivo com clipping
TTS_MAX_ZCR             = 0.65   # Mais permissivo com ruído
TTS_CHARS_PER_SECOND    = 15.0   # velocidade de fala estimada (chars/s) para PT-PT

# Reduzir timeouts para melhor responsividade
OLLAMA_REQUEST_TIMEOUT = 300  # segundos máximos por requisição
ANALYSIS_BATCH_SIZE = 3       # Processar 3 blocos por vez


# ─── VRAM / Gestão de Modelos ─────────────────────────────────────────────────
OLLAMA_UNLOAD_AFTER_ANALYSIS = True   # libertar VRAM do Ollama antes do TTS
OLLAMA_UNLOAD_URL_TEMPLATE   = "{base}/api/generate"   # usado no unload

# ─── Âncoras: Timeout e limite de tokens ─────────────────────────────────────
ANCHOR_TIMEOUT        = 90   # segundos máximos para gerar uma âncora (modelo frio demora mais)
ANCHOR_MAX_NEW_TOKENS = 800  # tokens de áudio máximos (~50s a 24Hz)

# ─── Síntese geral: limite de tokens por segmento ────────────────────────────
# A 24Hz, 1 token ≈ ~42ms de áudio.
# 2000 tokens ≈ ~83s — suficiente para qualquer segmento normal.
# Sem limite → modelo pode loopar e gerar minutos de ruído.
TTS_MAX_NEW_TOKENS    = 2000
TTS_BASE_TEMPERATURE = 0.27       # Temperatura base mais baixa (era 0.3)