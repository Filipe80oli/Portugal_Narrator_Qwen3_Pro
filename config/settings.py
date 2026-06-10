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

# Ollama
OLLAMA_BASE_URL = "http://localhost:11434"
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
