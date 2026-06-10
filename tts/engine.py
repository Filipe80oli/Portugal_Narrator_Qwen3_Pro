# tts/engine.py
# ─── Motor TTS: Qwen3-TTS Base (clonagem) + VoiceDesign (síntese) ─────────────

import re
import asyncio
import logging
import subprocess
from pathlib import Path

import torch

from config.settings import (
    QWEN3_MODEL_BASE, QWEN3_MODEL_VOICEDESIGN,
    NARRATOR_PT_PT_INSTRUCT, ANCHOR_TEXT
)

logger = logging.getLogger(__name__)


class TTSEngine:
    """Gere o carregamento lazy dos modelos Qwen3-TTS e a síntese de áudio."""

    def __init__(self, temp_dir: Path, log_fn=None):
        self.temp_dir = temp_dir
        self.log = log_fn or logger.info
        self.model_base = None       # Base  — clonagem de voz
        self.model_design = None     # VoiceDesign — síntese por descrição

    # ─── Carregamento de Modelos ──────────────────────────────────────────────

    def _load_model_sync(self, model_id: str, label: str):
        from qwen_tts import Qwen3TTSModel
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype  = torch.bfloat16 if device == 'cuda' else torch.float32
        attn   = 'sdpa' if device == 'cuda' else 'eager'
        self.log(f'   {label} -> {device.upper()} | attn={attn}')
        return Qwen3TTSModel.from_pretrained(
            model_id, device_map=device, dtype=dtype, attn_implementation=attn,
        )

    async def load_base(self):
        if self.model_base is not None:
            return
        self.log(f'🤖 A carregar Base (clonagem) -- {QWEN3_MODEL_BASE} ...')
        self.model_base = await asyncio.to_thread(
            self._load_model_sync, QWEN3_MODEL_BASE, 'Base'
        )
        self.log('✅ Modelo Base carregado.')

    async def load_voicedesign(self):
        if self.model_design is not None:
            return
        self.log(f'🤖 A carregar VoiceDesign -- {QWEN3_MODEL_VOICEDESIGN} ...')
        self.model_design = await asyncio.to_thread(
            self._load_model_sync, QWEN3_MODEL_VOICEDESIGN, 'VoiceDesign'
        )
        self.log('✅ Modelo VoiceDesign carregado.')

    # ─── Âncoras PT-PT ───────────────────────────────────────────────────────

    async def ensure_anchor(self, cid: str, cdata: dict):
        """Cria um ficheiro âncora para personagens sem .wav (garante voz PT-PT).
        
        Também guarda `ref_text` em cdata — obrigatório para o modo ICL do modelo Base.
        Para ficheiros .wav fornecidos pelo utilizador, ref_text fica vazio (modo x_vector).
        """
        anchor_path = self.temp_dir / f"anchor_{cid}.wav"

        # Já tem áudio e texto de referência → nada a fazer
        if cdata.get("ref_audio") and cdata.get("ref_text") is not None:
            return

        # Âncora gerada anteriormente em disco → apenas repor o estado
        if cdata.get("ref_audio") and anchor_path.exists() and str(anchor_path) == cdata.get("ref_audio"):
            if cdata.get("ref_text") is None:
                cdata["ref_text"] = ANCHOR_TEXT
            return

        # .wav fornecido pelo utilizador: usa x_vector_only (sem ref_text obrigatório)
        if cdata.get("ref_audio") and not anchor_path.exists():
            cdata.setdefault("ref_text", "")   # string vazia → x_vector_only_mode
            return

        # Gerar âncora do zero com VoiceDesign
        base_desc = cdata.get("description", "Voz neutra")
        if cid == "narrator":
            instruct = f"{NARRATOR_PT_PT_INSTRUCT} {base_desc}"
        else:
            instruct = f"{base_desc}. Sotaque de Portugal, português europeu."

        self.log(f"🇵🇹 A desenhar voz âncora PT-PT para '{cdata.get('name')}': {cid}...")
        try:
            wavs, sr = self.model_design.generate_voice_design(
                text=ANCHOR_TEXT, instruct=instruct,
                language='portuguese', temperature=0.1
            )
            self._write_audio(wavs, sr, str(anchor_path))
            cdata["ref_audio"] = str(anchor_path)
            cdata["ref_text"]  = ANCHOR_TEXT   # transcrição exata do áudio gerado
        except Exception as e:
            self.log(f"⚠️ Erro âncora PT-PT: {e}")

    # ─── Síntese ─────────────────────────────────────────────────────────────

    def synthesize(self, text: str, cdata: dict, emotion: str,
                   pace: float, out_path: str) -> bool:
        """Escolhe clone (Base) ou VoiceDesign conforme o personagem."""
        try:
            ref_audio = cdata.get('ref_audio')
            if ref_audio:
                self.clone_with_emotion(text, ref_audio, emotion, pace, out_path)
            else:
                base_desc = cdata.get('description', 'Voz neutra em português de Portugal')
                full_desc = base_desc
                if emotion and emotion != "neutral":
                    full_desc += f" Tom de voz {emotion}."
                self.generate_design(text, full_desc, emotion, out_path)
            return True
        except Exception as e:
            self.log(f'❌ Erro TTS: {e}')
            return False

    def clone_with_emotion(self, text: str, ref_audio: str,
                           emotion: str, pace: float, out_path: str,
                           ref_text: str = "") -> bool:
        """
        Clona a voz de ref_audio para sintetizar `text`.
        - ref_text não vazio  → modo ICL (melhor naturalidade, obrigatório para âncoras geradas)
        - ref_text vazio      → modo x_vector_only (para .wav externos sem transcrição)
        """
        try:
            clean = self._clean_text(text)
            kwargs = dict(
                text=clean,
                ref_audio=ref_audio,
                language='portuguese',
                temperature=0.3,
                top_p=0.95,
            )
            if ref_text:
                kwargs["ref_text"] = ref_text
            else:
                kwargs["x_vector_only_mode"] = True

            wavs, sr = self.model_base.generate_voice_clone(**kwargs)
            self._write_audio(wavs, sr, out_path)
            return True
        except Exception as e:
            self.log(f"❌ Erro na clonagem: {e}")
            return False

    def generate_clone(self, text: str, ref_audio: str, out_path: str,
                       ref_text: str = ""):
        kwargs = dict(
            text=text, ref_audio=ref_audio,
            language='portuguese', temperature=0.3, top_p=0.95,
        )
        if ref_text:
            kwargs["ref_text"] = ref_text
        else:
            kwargs["x_vector_only_mode"] = True
        wavs, sr = self.model_base.generate_voice_clone(**kwargs)
        self._write_audio(wavs, sr, out_path)

    def generate_design(self, text: str, description: str,
                        emotion: str, out_path: str):
        is_narrator = "narrator" in description.lower() or "narrador" in description.lower()
        gender_fix = "Voz masculina, homem de Portugal." if is_narrator else ""
        full_instruct = (
            f"{description}. {gender_fix} "
            "Sotaque de Lisboa, Portugal. Português Europeu. "
            "Pronúncia clara de Portugal, sem sotaque brasileiro."
        )
        wavs, sr = self.model_design.generate_voice_design(
            text=text, instruct=full_instruct,
            language='portuguese', temperature=0.3, top_p=0.95
        )
        self._write_audio(wavs, sr, out_path)

    # ─── Utilitários ─────────────────────────────────────────────────────────

    def _write_audio(self, wavs, sr: int, out_path: str):
        import soundfile as sf
        import numpy as np
        audio = wavs[0] if (hasattr(wavs, '__len__') and not isinstance(wavs, np.ndarray)) else wavs
        if hasattr(audio, 'cpu'):    audio = audio.cpu().numpy()
        if hasattr(audio, 'numpy'): audio = audio.numpy()
        sf.write(out_path, audio, sr)

    def _clean_text(self, text: str) -> str:
        text = text.replace('\u201c', '«').replace('\u201d', '»')
        text = text.replace('"', '«').replace('"', '»')
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        text = re.sub(r'[^\w\s«»\'\-.,;:!?…]', '', text, flags=re.UNICODE)
        if text and text[-1] not in '.!?…':
            text += '.'
        return text.strip()

    def create_silence(self, duration: float, filename: str) -> Path:
        path = self.temp_dir / filename
        if not path.exists():
            cmd = ['ffmpeg', '-y', '-f', 'lavfi', '-i', 'anullsrc=r=24000:cl=mono',
                   '-t', str(duration), '-c:a', 'pcm_s16le', str(path)]
            subprocess.run(cmd, capture_output=True, check=True)
        return path
