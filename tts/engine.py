# tts/engine.py
# ─── Motor TTS: Qwen3-TTS Base (clonagem) + VoiceDesign (síntese) ─────────────
#
# Melhorias v7.4:
#   • Validação de qualidade de áudio (RMS, ZCR, silêncio, duração)
#   • Retry automático com temperature escalante
#   • Carregamento seletivo de modelos (só carrega o necessário)
#   • Cache de segmentos: reutiliza WAVs já gerados em runs anteriores

import re
import asyncio
import logging
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

import torch

from config.settings import (
    QWEN3_MODEL_BASE, QWEN3_MODEL_VOICEDESIGN,
    NARRATOR_PT_PT_INSTRUCT, ANCHOR_TEXT,
    TTS_MAX_RETRIES, TTS_RETRY_TEMP_STEP,
    ANCHOR_TIMEOUT, ANCHOR_MAX_NEW_TOKENS,
    TTS_MAX_NEW_TOKENS,
)
from tts.audio_validator import validate_audio, log_quality
from tts.vram_manager import release_model, log_vram

logger = logging.getLogger(__name__)


class TTSEngine:
    """Gere o carregamento lazy dos modelos Qwen3-TTS e a síntese de áudio."""

    def __init__(self, temp_dir: Path, log_fn=None):
        self.temp_dir    = temp_dir
        self.log         = log_fn or logger.info
        self.model_base   = None   # Base  — clonagem de voz (ICL / x-vector)
        self.model_design = None   # VoiceDesign — síntese por descrição

    # ═══════════════════════════════════════════════════════════════════════════
    # Carregamento / Libertação de Modelos
    # ═══════════════════════════════════════════════════════════════════════════

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
        log_vram(self.log)
        self.log(f'🤖 A carregar Base (clonagem) -- {QWEN3_MODEL_BASE} ...')
        self.model_base = await asyncio.to_thread(
            self._load_model_sync, QWEN3_MODEL_BASE, 'Base'
        )
        log_vram(self.log)
        self.log('✅ Modelo Base carregado.')

    async def load_voicedesign(self):
        if self.model_design is not None:
            return
        log_vram(self.log)
        self.log(f'🤖 A carregar VoiceDesign -- {QWEN3_MODEL_VOICEDESIGN} ...')
        self.model_design = await asyncio.to_thread(
            self._load_model_sync, QWEN3_MODEL_VOICEDESIGN, 'VoiceDesign'
        )
        log_vram(self.log)
        self.log('✅ Modelo VoiceDesign carregado.')

    def release_base(self):
        release_model("model_base", self, self.log)

    def release_voicedesign(self):
        release_model("model_design", self, self.log)

    # ═══════════════════════════════════════════════════════════════════════════
    # Carregamento Seletivo (análise dos segmentos)
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def needs_base(characters: dict) -> bool:
        """True se pelo menos um personagem tem ref_audio (clonagem)."""
        return any(c.get("ref_audio") for c in characters.values())

    @staticmethod
    def needs_voicedesign(characters: dict) -> bool:
        """True se pelo menos um personagem NÃO tem ref_audio (VoiceDesign)."""
        return any(not c.get("ref_audio") for c in characters.values())

    # ═══════════════════════════════════════════════════════════════════════════
    # Cache de Segmentos
    # ═══════════════════════════════════════════════════════════════════════════

    def segment_cache_path(self, seg_index: int) -> Path:
        return self.temp_dir / f"seg_{seg_index:05d}.wav"

    def is_segment_cached(self, seg_index: int, text: str) -> bool:
        """
        Verifica se já existe um WAV válido para este segmento.
        Valida com o audio_validator para não reutilizar ficheiros corrompidos.
        """
        path = self.segment_cache_path(seg_index)
        if not path.exists():
            return False
        q = validate_audio(str(path), text)
        if q.ok:
            return True
        # Ficheiro existe mas está corrompido — apagar para gerar de novo
        path.unlink(missing_ok=True)
        return False

    def count_cached_segments(self, segments: list) -> int:
        """Conta quantos segmentos já têm WAV válido em cache."""
        count = 0
        for i, seg in enumerate(segments):
            if isinstance(seg, dict) and seg.get("text"):
                if self.is_segment_cached(i, seg["text"]):
                    count += 1
        return count

    # ═══════════════════════════════════════════════════════════════════════════
    # Âncoras PT-PT
    # ═══════════════════════════════════════════════════════════════════════════


    def _generate_anchor_sync(self, cid: str, instruct: str,
                               anchor_path: str, cdata: dict) -> bool:
        """
        Gera a âncora de voz num executor com timeout real.
        Retorna True se gerou com sucesso, False em timeout ou erro.
        """
        def _do_generate():
            wavs, sr = self.model_design.generate_voice_design(
                text=ANCHOR_TEXT,
                instruct=instruct,
                language="portuguese",
                temperature=0.3,          # ligeiramente acima de 0.1 para evitar loops
                top_p=0.90,
                max_new_tokens=ANCHOR_MAX_NEW_TOKENS,
            )
            self._write_audio(wavs, sr, anchor_path)

        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_do_generate)
            try:
                fut.result(timeout=ANCHOR_TIMEOUT)
                cdata["ref_audio"] = anchor_path
                cdata["ref_text"]  = ANCHOR_TEXT
                self.log(f"   ✅ Âncora gerada: {cid}")
                return True
            except FuturesTimeout:
                self.log(f"   ⏰ Timeout ({ANCHOR_TIMEOUT}s) ao gerar âncora para {cid}.")
                fut.cancel()
                return False
            except Exception as e:
                self.log(f"   ❌ Erro ao gerar âncora para {cid}: {e}")
                return False

    async def ensure_anchor(self, cid: str, cdata: dict):
        """
        Cria âncora de voz PT-PT (VoiceDesign) para personagens sem .wav externo.
        Guarda ref_text para o modo ICL do modelo Base.
        """
        anchor_path = self.temp_dir / f"anchor_{cid}.wav"

        # Já tem áudio e texto de referência → nada a fazer
        if cdata.get("ref_audio") and cdata.get("ref_text") is not None:
            return

        # Âncora gerada anteriormente em disco → repor estado
        if cdata.get("ref_audio") and anchor_path.exists() and \
                str(anchor_path) == cdata.get("ref_audio"):
            if cdata.get("ref_text") is None:
                cdata["ref_text"] = ANCHOR_TEXT
            return

        # .wav externo fornecido pelo utilizador → x_vector_only
        if cdata.get("ref_audio") and not anchor_path.exists():
            cdata.setdefault("ref_text", "")
            return

        # Gerar âncora com VoiceDesign
        base_desc = cdata.get("description", "Voz neutra")
        if cid == "narrator":
            instruct = f"{NARRATOR_PT_PT_INSTRUCT} {base_desc}"
        else:
            instruct = f"{base_desc}. Sotaque de Portugal, português europeu."

        self.log(f"🇵🇹 A desenhar voz âncora PT-PT para '{cdata.get('name')}': {cid}...")
        self.log(f"   ⏱️ Timeout: {ANCHOR_TIMEOUT}s — se demorar mais, usará VoiceDesign direto.")

        # VoiceDesign precisa de estar carregado para gerar âncoras
        if self.model_design is None:
            await self.load_voicedesign()

        success = await asyncio.to_thread(
            self._generate_anchor_sync,
            cid, instruct, str(anchor_path), cdata
        )
        if not success:
            # Fallback: sem âncora — síntese por VoiceDesign direta em cada segmento
            self.log(f"   ⚠️ Âncora falhou — '{cdata.get('name')}' usará VoiceDesign por segmento.")
            cdata["ref_audio"] = None
            cdata["ref_text"]  = None

    # ═══════════════════════════════════════════════════════════════════════════
    # Síntese com Retry + Validação de Qualidade
    # ═══════════════════════════════════════════════════════════════════════════

    def clone_with_emotion(self, text: str, ref_audio: str | None,
                           emotion: str, pace: float, out_path: str,
                           ref_text: str = "",
                           voice_description: str = "") -> bool:
        """
        Clona voz com retry automático.
        Se ref_audio for None (âncora falhou/timeout), usa VoiceDesign com
        voice_description em vez de bloquear ou gerar silêncio.
        """
        # ── Sem âncora: usar VoiceDesign directamente ─────────────────────────
        if not ref_audio:
            desc = voice_description or "Voz neutra, português de Portugal, sotaque de Lisboa."
            return self.generate_design(text, desc, emotion, out_path)

        base_temp = 0.3

        for attempt in range(1, TTS_MAX_RETRIES + 1):
            temp = min(base_temp + (attempt - 1) * TTS_RETRY_TEMP_STEP, 0.85)

            try:
                clean  = self._clean_text(text)
                kwargs = dict(
                    text=clean,
                    ref_audio=ref_audio,
                    language='portuguese',
                    temperature=temp,
                    top_p=0.95,
                    max_new_tokens=TTS_MAX_NEW_TOKENS,
                )
                if ref_text:
                    kwargs["ref_text"] = ref_text
                else:
                    kwargs["x_vector_only_mode"] = True

                wavs, sr = self.model_base.generate_voice_clone(**kwargs)
                self._write_audio(wavs, sr, out_path)

            except Exception as e:
                self.log(f"   ❌ Tentativa {attempt}/{TTS_MAX_RETRIES} excepção: {e}")
                if attempt == TTS_MAX_RETRIES:
                    return False
                continue

            # Validar qualidade
            q = validate_audio(out_path, text)
            if q.ok:
                if attempt > 1:
                    self.log(f"   ✅ Qualidade OK na tentativa {attempt} "
                             f"(temp={temp:.2f}) — {q}")
                return True

            self.log(f"   ⚠️ Qualidade baixa tentativa {attempt}/{TTS_MAX_RETRIES}: "
                     f"{q.reason}  {q}")

            if attempt < TTS_MAX_RETRIES:
                # Estratégia de recuperação
                strategy = _retry_strategy(attempt, ref_text)
                self.log(f"   🔄 Estratégia: {strategy}")

                if strategy == "split" and len(text) > 80:
                    # Dividir em partes e tentar de novo na próxima iteração
                    text = _split_text_for_retry(text)
                elif strategy == "x_vector" and ref_text:
                    # Fallback para x_vector_only
                    ref_text = ""

        return False   # Esgotadas as tentativas

    def generate_clone(self, text: str, ref_audio: str, out_path: str,
                       ref_text: str = "") -> bool:
        return self.clone_with_emotion(
            text, ref_audio, "neutral", 1.0, out_path, ref_text
        )

    def generate_design(self, text: str, description: str,
                        emotion: str, out_path: str) -> bool:
        base_temp = 0.3

        is_narrator = "narrator" in description.lower() or "narrador" in description.lower()
        gender_fix  = "Voz masculina, homem de Portugal." if is_narrator else ""
        full_instruct = (
            f"{description}. {gender_fix} "
            "Sotaque de Lisboa, Portugal. Português Europeu. "
            "Pronúncia clara de Portugal, sem sotaque brasileiro."
        )

        for attempt in range(1, TTS_MAX_RETRIES + 1):
            temp = min(base_temp + (attempt - 1) * TTS_RETRY_TEMP_STEP, 0.85)
            try:
                wavs, sr = self.model_design.generate_voice_design(
                    text=text, instruct=full_instruct,
                    language='portuguese', temperature=temp, top_p=0.95,
                    max_new_tokens=TTS_MAX_NEW_TOKENS,
                )
                self._write_audio(wavs, sr, out_path)
            except Exception as e:
                self.log(f"   ❌ VoiceDesign tentativa {attempt}: {e}")
                if attempt == TTS_MAX_RETRIES:
                    return False
                continue

            q = validate_audio(out_path, text)
            if q.ok:
                if attempt > 1:
                    self.log(f"   ✅ VoiceDesign OK na tentativa {attempt}")
                return True
            self.log(f"   ⚠️ VoiceDesign qualidade baixa tentativa {attempt}: {q.reason}")

        return False

    # ═══════════════════════════════════════════════════════════════════════════
    # Utilitários
    # ═══════════════════════════════════════════════════════════════════════════

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


# ─── Helpers de retry ─────────────────────────────────────────────────────────

def _retry_strategy(attempt: int, ref_text: str) -> str:
    """Define estratégia para cada tentativa."""
    if attempt == 1:
        return "temperature_up"    # apenas aumenta temperature
    if attempt == 2 and ref_text:
        return "x_vector"          # 3ª tentativa: abandona ICL, usa x_vector
    return "split"                 # última: dividir texto


def _split_text_for_retry(text: str) -> str:
    """
    Se o texto for muito longo, encurta para o primeiro período completo.
    O modelo TTS degrada com textos >200 caracteres.
    """
    # Procura o primeiro ponto final, !, ? antes dos 200 chars
    for i, ch in enumerate(text[:200]):
        if ch in '.!?' and i > 30:
            return text[:i+1].strip()
    return text[:150].strip() + '.'
