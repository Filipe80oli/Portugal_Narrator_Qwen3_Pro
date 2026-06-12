# tts/audio_validator.py
# ─── Validação de qualidade de áudio gerado pelo TTS ─────────────────────────
#
# Deteta 4 falhas comuns dos modelos TTS:
#   1. Ficheiro vazio ou demasiado curto para o texto
#   2. Silêncio excessivo (modelo "alucinouu" silêncio)
#   3. Amplitude RMS fora do intervalo esperado (ruído / clipping)
#   4. Zero-Crossing Rate elevada (ruído branco, língua incompreensível)

import logging
import numpy as np
import soundfile as sf
from pathlib import Path

from config.settings import (
    TTS_MIN_DURATION_RATIO,
    TTS_MAX_SILENCE_RATIO,
    TTS_MIN_RMS,
    TTS_MAX_RMS,
    TTS_MAX_ZCR,
    TTS_CHARS_PER_SECOND,
)

logger = logging.getLogger(__name__)

# Resultado da validação
class AudioQuality:
    __slots__ = ("ok", "reason", "rms", "zcr", "duration",
                 "silence_ratio", "expected_min_duration")

    def __init__(self, ok: bool, reason: str = "",
                 rms: float = 0.0, zcr: float = 0.0,
                 duration: float = 0.0, silence_ratio: float = 0.0,
                 expected_min: float = 0.0):
        self.ok                   = ok
        self.reason               = reason
        self.rms                  = rms
        self.zcr                  = zcr
        self.duration             = duration
        self.silence_ratio        = silence_ratio
        self.expected_min_duration = expected_min

    def __str__(self):
        if self.ok:
            return (f"OK  dur={self.duration:.2f}s  "
                    f"rms={self.rms:.4f}  zcr={self.zcr:.4f}  "
                    f"sil={self.silence_ratio:.2f}")
        return (f"FAIL [{self.reason}]  dur={self.duration:.2f}s  "
                f"rms={self.rms:.4f}  zcr={self.zcr:.4f}  "
                f"sil={self.silence_ratio:.2f}")


def validate_audio(wav_path: str, text: str) -> AudioQuality:
    """
    Valida o ficheiro WAV gerado.
    Retorna AudioQuality com .ok=True se passar todos os testes.
    """
    path = Path(wav_path)

    # ── 1. Ficheiro existe e tem tamanho razoável ─────────────────────────────
    if not path.exists() or path.stat().st_size < 1024:
        return AudioQuality(False, "ficheiro_vazio_ou_ausente")

    try:
        audio, sr = sf.read(str(path), dtype='float32', always_2d=False)
    except Exception as e:
        return AudioQuality(False, f"erro_leitura:{e}")

    # Mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    n_samples = len(audio)
    duration  = n_samples / sr

    # ── 2. Duração mínima esperada ────────────────────────────────────────────
    expected_min = max(0.5, len(text) / TTS_CHARS_PER_SECOND * TTS_MIN_DURATION_RATIO)
    if duration < expected_min:
        return AudioQuality(False, "duracao_insuficiente",
                            duration=duration, expected_min=expected_min)

    # ── 3. RMS global ─────────────────────────────────────────────────────────
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < TTS_MIN_RMS:
        return AudioQuality(False, "rms_muito_baixo_silencio", rms=rms, duration=duration)
    if rms > TTS_MAX_RMS:
        return AudioQuality(False, "rms_muito_alto_clipping", rms=rms, duration=duration)

    # ── 4. Rácio de silêncio ──────────────────────────────────────────────────
    # Frame 20ms para análise de energia
    frame_len  = int(sr * 0.02)
    frames     = [audio[i:i+frame_len] for i in range(0, n_samples - frame_len, frame_len)]
    if frames:
        frame_rms    = np.array([np.sqrt(np.mean(f**2)) for f in frames])
        silence_ratio = float(np.mean(frame_rms < TTS_MIN_RMS))
        if silence_ratio > TTS_MAX_SILENCE_RATIO:
            return AudioQuality(False, "silencio_excessivo",
                                rms=rms, duration=duration,
                                silence_ratio=silence_ratio)
    else:
        silence_ratio = 0.0

    # ── 5. Zero-Crossing Rate (ruído / língua incompreensível) ────────────────
    # Calculado apenas na parte activa (acima do limiar RMS)
    active_mask  = frame_rms >= TTS_MIN_RMS
    if active_mask.any():
        active_audio = np.concatenate([
            frames[j] for j in range(len(frames)) if active_mask[j]
        ])
        zcr = float(np.mean(np.abs(np.diff(np.sign(active_audio)))) / 2)
    else:
        zcr = 0.0

    if zcr > TTS_MAX_ZCR:
        return AudioQuality(False, "zcr_elevado_ruido_provavel",
                            rms=rms, zcr=zcr, duration=duration,
                            silence_ratio=silence_ratio)

    return AudioQuality(True, rms=rms, zcr=zcr, duration=duration,
                        silence_ratio=silence_ratio)


def log_quality(q: AudioQuality, seg_index: int, log_fn) -> None:
    icon = "✅" if q.ok else "⚠️"
    log_fn(f"   {icon} QA[{seg_index}]: {q}")
