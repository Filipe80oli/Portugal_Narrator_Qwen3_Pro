# cinema/mixer.py
# ─── Mixagem Cinema: voz + efeitos sonoros + música via FFmpeg ────────────────

import logging
import subprocess
from pathlib import Path

from config.settings import (
    CINEMA_VOICE_VOL, CINEMA_SFX_VOL, CINEMA_MUSIC_VOL,
    CINEMA_SFX_DEFAULT_DURATION, CINEMA_MUSIC_DEFAULT_DURATION,
)
from cinema.sound_db import get_db

logger = logging.getLogger(__name__)


def mix_segment_with_sfx(
    voice_wav: str,
    sound_events: list,       # lista de dicts de sound_analyzer
    temp_dir: Path,
    seg_index: int,
    log_fn=None,
) -> str:
    """
    Mistura um segmento de voz com os efeitos sonoros detetados.
    Retorna o caminho do WAV mixado (ou o voice_wav original se sem sons).
    """
    log = log_fn or logger.info
    db = get_db()

    if not sound_events:
        return voice_wav

    # Filtrar sons encontrados na DB
    resolved = []
    for ev in sound_events:
        path = db.find(ev["sound"])
        if path:
            resolved.append({**ev, "path": str(path)})
        else:
            log(f"   ⚠️ Som não encontrado na DB: '{ev['sound']}'")

    if not resolved:
        return voice_wav

    out_path = str(temp_dir / f"seg_{seg_index:05d}_mix.wav")

    # Construir comando ffmpeg com amix
    # Input 0: voz; Inputs 1..N: efeitos
    cmd = ["ffmpeg", "-y", "-i", voice_wav]
    filter_parts = []
    sfx_labels   = []

    for j, ev in enumerate(resolved):
        pos      = ev.get("position", "during")
        duration = min(ev.get("duration", CINEMA_SFX_DEFAULT_DURATION), 10.0)
        vol_db   = ev.get("volume", CINEMA_SFX_VOL)

        # Ajuste de volume + limite de duração por evento
        cmd += ["-i", ev["path"]]
        label = f"sfx{j}"
        filter_parts.append(
            f"[{j+1}:a]"
            f"atrim=duration={duration},"
            f"volume={vol_db}dB,"
            f"apad=pad_dur=0"
            f"[{label}]"
        )
        if pos == "before":
            # Atraso negativo não é possível; adiciona silêncio antes da voz
            sfx_labels.append(f"[{label}]")
        else:
            sfx_labels.append(f"[{label}]")

    # Mix final: voz + todos os SFX
    n_inputs = 1 + len(sfx_labels)
    mix_inputs = "[0:a]" + "".join(sfx_labels)
    filter_parts.append(
        f"{mix_inputs}amix=inputs={n_inputs}:duration=longest:dropout_transition=2[out]"
    )

    filter_complex = ";".join(filter_parts)
    cmd += ["-filter_complex", filter_complex, "-map", "[out]",
            "-c:a", "pcm_s16le", "-ar", "24000", out_path]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        err = result.stderr.decode(errors="replace")
        log(f"   ⚠️ FFmpeg SFX error: {err[-200:]}")
        return voice_wav  # fallback para voz limpa

    return out_path


def mix_music_under_segment(
    voice_wav: str,
    music_event: dict,
    temp_dir: Path,
    seg_index: int,
    log_fn=None,
) -> str:
    """
    Adiciona música de fundo sob um segmento de voz.
    Retorna o caminho do WAV resultante.
    """
    log = log_fn or logger.info
    db = get_db()

    music_key  = music_event.get("music", "")
    music_path = db.find(f"music_{music_key}") or db.find(music_key)

    if not music_path:
        log(f"   ⚠️ Música não encontrada na DB: '{music_key}'")
        return voice_wav

    vol_db   = music_event.get("music_volume", CINEMA_MUSIC_VOL)
    duration = music_event.get("music_duration", CINEMA_MUSIC_DEFAULT_DURATION)
    out_path = str(temp_dir / f"seg_{seg_index:05d}_music.wav")

    cmd = [
        "ffmpeg", "-y",
        "-i", voice_wav,
        "-i", str(music_path),
        "-filter_complex",
        f"[1:a]volume={vol_db}dB,atrim=duration={duration},aloop=loop=-1:size=44100[music];"
        f"[0:a][music]amix=inputs=2:duration=first:dropout_transition=3[out]",
        "-map", "[out]",
        "-c:a", "pcm_s16le", "-ar", "24000",
        out_path
    ]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        err = result.stderr.decode(errors="replace")
        log(f"   ⚠️ FFmpeg Music error: {err[-200:]}")
        return voice_wav

    return out_path


def apply_cinema_mix(
    voice_wav: str,
    sound_data: dict,   # {"sounds": [...], "music": {...}|None}
    temp_dir: Path,
    seg_index: int,
    log_fn=None,
) -> str:
    """
    Pipeline completo de mixagem Cinema para um segmento:
    1. Mistura efeitos sonoros
    2. Adiciona música de fundo
    Retorna o caminho do WAV final.
    """
    current = voice_wav

    if sound_data.get("sounds"):
        current = mix_segment_with_sfx(
            current, sound_data["sounds"], temp_dir, seg_index, log_fn
        )

    if sound_data.get("music"):
        current = mix_music_under_segment(
            current, sound_data["music"], temp_dir, seg_index, log_fn
        )

    return current
