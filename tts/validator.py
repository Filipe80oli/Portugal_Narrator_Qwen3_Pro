# tts/validator.py
# ─── Validação de qualidade do áudio gerado ───────────────────────────────────
#
# Deteta os dois problemas mais comuns do Qwen3-TTS:
#   1. Ruído / alucinação sonora  → energia anormalmente alta ou espetro plano
#   2. Língua incompreensível     → segmentos de silêncio excessivo ou duração absurda
#
# Usa apenas numpy + scipy (já dependências comuns); sem modelos externos.

import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# ─── Limites configuráveis ────────────────────────────────────────────────────

# Duração mínima/máxima aceitável por palavra do texto (segundos)
MIN_SECS_PER_WORD = 0.08   # muito rápido → provável lixo
MAX_SECS_PER_WORD = 1.8    # muito lento  → provável silêncio/falha

# Rácio máximo de silêncio no áudio (amostras abaixo de threshold)
MAX_SILENCE_RATIO = 0.75   # >75% silêncio → falha

# Energia RMS mínima (áudio demasiado fraco → pode ser ruído de baixo nível)
MIN_RMS = 0.002

# Espetro: desvio padrão mínimo dos bins de frequência normalizados
# (espetro muito plano = ruído branco / tom constante)
MIN_SPECTRAL_STD = 0.008

# Limite de energia para detetar silêncio por amostra
SILENCE_THRESHOLD = 0.01

# Comprimento mínimo do áudio em segundos
MIN_DURATION_SECS = 0.3


class AudioQuality:
    """Resultado da validação com diagnóstico detalhado."""

    def __init__(self):
        self.ok            = True
        self.reasons: list[str] = []
        self.duration      = 0.0
        self.rms           = 0.0
        self.silence_ratio = 0.0
        self.spectral_std  = 0.0
        self.secs_per_word = 0.0

    def fail(self, reason: str):
        self.ok = False
        self.reasons.append(reason)

    def summary(self) -> str:
        if self.ok:
            return (f"✅ OK  dur={self.duration:.1f}s  rms={self.rms:.4f}"
                    f"  sil={self.silence_ratio:.0%}  spk={self.spectral_std:.4f}")
        return "❌ " + " | ".join(self.reasons)


def validate_audio(wav_path: str, source_text: str = "") -> AudioQuality:
    """
    Valida um ficheiro WAV gerado pelo TTS.
    
    Parâmetros
    ----------
    wav_path    : caminho para o .wav gerado
    source_text : texto que foi sintetizado (usado para calcular secs_per_word)

    Retorna AudioQuality com .ok = True se o áudio parece válido.
    """
    q = AudioQuality()
    path = Path(wav_path)

    # ── Ficheiro existe e não está vazio ─────────────────────────────────────
    if not path.exists() or path.stat().st_size < 1000:
        q.fail("ficheiro ausente ou demasiado pequeno")
        return q

    # ── Carregar áudio ───────────────────────────────────────────────────────
    try:
        import soundfile as sf
        audio, sr = sf.read(str(path), dtype='float32', always_2d=False)
    except Exception as e:
        q.fail(f"erro ao ler WAV: {e}")
        return q

    if audio.ndim > 1:
        audio = audio.mean(axis=1)   # stereo → mono

    n_samples = len(audio)
    q.duration = n_samples / sr

    # ── Duração mínima ───────────────────────────────────────────────────────
    if q.duration < MIN_DURATION_SECS:
        q.fail(f"duração demasiado curta ({q.duration:.2f}s)")
        return q

    # ── RMS (energia média) ──────────────────────────────────────────────────
    q.rms = float(np.sqrt(np.mean(audio ** 2)))
    if q.rms < MIN_RMS:
        q.fail(f"energia RMS muito baixa ({q.rms:.5f}) — provável silêncio")

    # ── Rácio de silêncio ────────────────────────────────────────────────────
    silent_samples = np.sum(np.abs(audio) < SILENCE_THRESHOLD)
    q.silence_ratio = float(silent_samples / n_samples)
    if q.silence_ratio > MAX_SILENCE_RATIO:
        q.fail(f"silêncio excessivo ({q.silence_ratio:.0%})")

    # ── Espetro de frequências (deteta ruído branco / tom constante) ─────────
    try:
        # FFT em janela de 2048 amostras no meio do áudio
        mid   = n_samples // 2
        chunk = audio[max(0, mid - 1024): mid + 1024]
        if len(chunk) >= 512:
            spectrum = np.abs(np.fft.rfft(chunk, n=2048))
            # Normalizar e calcular desvio padrão
            spec_norm = spectrum / (spectrum.max() + 1e-9)
            q.spectral_std = float(np.std(spec_norm))
            if q.spectral_std < MIN_SPECTRAL_STD:
                q.fail(f"espetro anormalmente plano ({q.spectral_std:.5f}) — provável ruído")
    except Exception:
        pass   # análise espetral é opcional

    # ── Ritmo: segundos por palavra ──────────────────────────────────────────
    if source_text:
        n_words = max(1, len(source_text.split()))
        q.secs_per_word = q.duration / n_words
        if q.secs_per_word < MIN_SECS_PER_WORD:
            q.fail(f"demasiado rápido ({q.secs_per_word:.3f}s/palavra) — provável alucinação")
        elif q.secs_per_word > MAX_SECS_PER_WORD:
            q.fail(f"demasiado lento ({q.secs_per_word:.3f}s/palavra) — provável silêncio")

    return q


def validate_audio_strict(wav_path: str, source_text: str = "") -> AudioQuality:
    """
    Versão mais permissiva para textos muito curtos (< 5 palavras).
    Aplica apenas os testes de ficheiro, RMS e silêncio.
    """
    q = validate_audio(wav_path, source_text)
    # Para textos curtos, ignorar o teste de ritmo
    if source_text and len(source_text.split()) < 5:
        q.reasons = [r for r in q.reasons if "s/palavra" not in r]
        q.ok = len(q.reasons) == 0
    return q
