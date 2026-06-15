# config/settings.py
# ─── Configurações globais da aplicação ───────────────────────────────────────

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

PTPT_ACCENT_SUFFIX = (
    " Sotaque de Portugal continental, português europeu. "
    "Nunca brasileiro. Vogais fechadas, dicção clara de Lisboa."
)

ANCHOR_TEXT = (
    "Estou a falar com o sotaque de Lisboa, em Portugal. "
    "Esta é a minha voz europeia."
)

# ─── Modos de Produção ────────────────────────────────────────────────────────
PRODUCTION_MODES    = ["🎙️ Narrador", "🎭 Novela", "🎬 Cinema"]
PRODUCTION_MODE_IDS = ["narrator",    "novela",    "cinema"]

# ─── Cinema: volumes de mixagem ───────────────────────────────────────────────
CINEMA_VOICE_VOL              =   0
CINEMA_SFX_VOL                =  -6
CINEMA_MUSIC_VOL              = -18
CINEMA_SFX_DEFAULT_DURATION   = 2.0
CINEMA_MUSIC_DEFAULT_DURATION = 8.0

# ─── Categorias de sons ───────────────────────────────────────────────────────
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
TTS_MAX_RETRIES        = 4
TTS_RETRY_TEMP_STEP    = 0.05
TTS_MIN_DURATION_RATIO = 0.40    # era 0.05 — exige duração proporcional ao texto
TTS_MAX_SILENCE_RATIO  = 0.85    # era 0.90
TTS_MIN_RMS            = 0.001   # era 0.0005 — chiado Qwen tem RMS ~0.002
TTS_MAX_RMS            = 0.95
TTS_MAX_ZCR            = 0.40    # era 0.65 — ruído branco: 0.45-0.60, fala: 0.10-0.35
TTS_CHARS_PER_SECOND   = 15.0

# ─── Timeouts ─────────────────────────────────────────────────────────────────
OLLAMA_REQUEST_TIMEOUT = 300
ANALYSIS_BATCH_SIZE    = 3

# ─── VRAM / Gestão de Modelos ─────────────────────────────────────────────────
OLLAMA_UNLOAD_AFTER_ANALYSIS = True
OLLAMA_UNLOAD_URL_TEMPLATE   = "{base}/api/generate"

# ─── Âncoras ──────────────────────────────────────────────────────────────────
ANCHOR_TIMEOUT        = 60     # era 90 — âncora de 10s não precisa de mais
ANCHOR_MAX_NEW_TOKENS = 260    # era 800 — 10s × (1/0.042ms) ≈ 238 tokens
ANCHOR_MAX_RETRIES     = 8   # tentativas para âncoras — mais importante acertar

# ─── Síntese geral ────────────────────────────────────────────────────────────
# A 24Hz, 1 token ≈ ~42ms. 2000 tokens ≈ ~83s.
TTS_MAX_NEW_TOKENS   = 2000
TTS_BASE_TEMPERATURE = 0.15    # era 0.27 — mais baixo = menos ruído