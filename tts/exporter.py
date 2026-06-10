# tts/exporter.py
# ─── Exportação do audiobook final para M4B via FFmpeg ────────────────────────

import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def create_m4b(files: list, title: str, author: str,
               cover_path: str = "", log_fn=None) -> bool:
    """
    Concatena os ficheiros WAV e exporta como .m4b com metadados.
    Retorna True em sucesso, False em falha.
    """
    log = log_fn or logger.info
    log("📦 A criar M4B final...")

    list_file = Path("concat_list.txt")
    output = f"{title}.m4b"

    try:
        with open(list_file, "w", encoding="utf-8") as f:
            for fp in files:
                f.write(f"file '{Path(fp).resolve().as_posix()}'\n")

        cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', str(list_file)]

        if cover_path:
            cmd.extend([
                '-i', cover_path,
                '-map', '0:a', '-map', '1:v',
                '-c:v', 'mjpeg',
                '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                '-disposition:v:0', 'attached_pic'
            ])
        else:
            cmd.extend(['-map', '0:a'])

        cmd.extend([
            '-c:a', 'aac', '-b:a', '96k', '-ar', '24000',
            '-metadata', f'title={title}',
            '-metadata', f'artist={author}',
            output
        ])

        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            err = result.stderr.decode(errors='replace')
            log(f"❌ ffmpeg: {err[-200:]}")
            return False

        log(f"✨ CONCLUÍDO: {output}")
        return True

    except Exception as e:
        log(f"❌ Erro M4B: {e}")
        return False
    finally:
        if list_file.exists():
            list_file.unlink()
