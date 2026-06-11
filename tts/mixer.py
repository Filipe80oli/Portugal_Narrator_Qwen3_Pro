# tts/mixer.py
# ─── Mixagem de voz + efeitos sonoros + música (modo Cinema) ──────────────────
#
# Usa FFmpeg para:
#   1. Redimensionar cada som ao tempo necessário (loop ou trim)
#   2. Aplicar volume em dB
#   3. Misturar tudo com a pista de voz
#
# Dependências: FFmpeg no PATH

import subprocess
import logging
import tempfile
from pathlib import Path

from core.sound_db import get_sound_path

logger = logging.getLogger(__name__)


def mix_segment_cinema(voice_wav: str, sfx_list: list, music: dict | None,
                        out_path: str, temp_dir: Path) -> bool:
    """
    Mistura a pista de voz com efeitos sonoros e música de fundo.

    voice_wav  : caminho do WAV da voz sintetizada
    sfx_list   : lista de dicts {"sound", "offset_ms", "duration_s", "volume_db"}
    music      : dict {"sound", "duration_s", "volume_db"} ou None
    out_path   : caminho de saída
    temp_dir   : pasta para ficheiros temporários
    """
    # Se não há nada para misturar, copia apenas
    if not sfx_list and not music:
        import shutil
        shutil.copy2(voice_wav, out_path)
        return True

    try:
        # Obter duração da voz
        voice_dur = _get_duration(voice_wav)
        if voice_dur <= 0:
            import shutil
            shutil.copy2(voice_wav, out_path)
            return True

        # Construir comando FFmpeg de mixagem
        inputs  = ["-i", voice_wav]          # input 0 = voz
        filters = []
        mix_labels = ["[0:a]"]               # a voz entra direta no amix

        stream_idx = 1  # próximo índice de input

        # ── Efeitos Sonoros ──────────────────────────────────────────────────
        for sfx in sfx_list:
            sfx_path = get_sound_path(sfx["sound"])
            if not sfx_path:
                logger.debug(f"SFX não encontrado: {sfx['sound']}")
                continue

            offset_s   = sfx["offset_ms"] / 1000.0
            duration_s = sfx["duration_s"]
            volume_db  = sfx["volume_db"]
            label      = f"[sfx{stream_idx}]"

            inputs += ["-i", str(sfx_path)]
            # trim + aloop para garantir duração, adelay para offset, volume
            filters.append(
                f"[{stream_idx}:a]"
                f"aloop=loop=-1:size=44100,"
                f"atrim=duration={duration_s:.3f},"
                f"adelay={int(offset_s*1000)}|{int(offset_s*1000)},"
                f"volume={volume_db}dB"
                f"{label}"
            )
            mix_labels.append(label)
            stream_idx += 1

        # ── Música de fundo ──────────────────────────────────────────────────
        if music:
            music_path = get_sound_path(music["sound"])
            if music_path:
                duration_s = min(music["duration_s"], voice_dur + 1.0)
                volume_db  = music["volume_db"]
                label      = f"[music{stream_idx}]"

                inputs += ["-i", str(music_path)]
                filters.append(
                    f"[{stream_idx}:a]"
                    f"aloop=loop=-1:size=44100,"
                    f"atrim=duration={duration_s:.3f},"
                    f"afade=t=out:st={max(0, duration_s-1.5):.3f}:d=1.5,"
                    f"volume={volume_db}dB"
                    f"{label}"
                )
                mix_labels.append(label)
                stream_idx += 1

        # Se só ficou a voz (todos os sons em falta), copia
        if len(mix_labels) == 1:
            import shutil
            shutil.copy2(voice_wav, out_path)
            return True

        # ── amix final ───────────────────────────────────────────────────────
        n = len(mix_labels)
        filter_str = "".join(f"{lbl}" for lbl in mix_labels)
        filters.append(f"{''.join(mix_labels)}amix=inputs={n}:duration=first:dropout_transition=2[out]")

        filter_complex = ";".join(filters)

        cmd = (
            ["ffmpeg", "-y"]
            + inputs
            + ["-filter_complex", filter_complex,
               "-map", "[out]",
               "-c:a", "pcm_s16le", "-ar", "24000", "-ac", "1",
               str(out_path)]
        )

        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            err = result.stderr.decode(errors="replace")[-400:]
            logger.error(f"FFmpeg mix erro: {err}")
            # Fallback: só voz
            import shutil
            shutil.copy2(voice_wav, out_path)
            return True   # não falha a geração

        return True

    except Exception as e:
        logger.error(f"Erro mixer: {e}")
        import shutil
        try:
            shutil.copy2(voice_wav, out_path)
        except Exception:
            pass
        return True  # continua mesmo com erro


def _get_duration(wav_path: str) -> float:
    """Retorna a duração de um WAV em segundos usando ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", wav_path],
            capture_output=True, timeout=10
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0
