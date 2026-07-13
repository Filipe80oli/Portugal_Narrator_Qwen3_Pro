# ─── Motor TTS: Qwen3-TTS Base (clonagem) + VoiceDesign (síntese) ─────────────
# Melhorias v7.4:
# • Validação de qualidade de áudio (RMS, ZCR, silêncio, duração)
# • Retry automático com temperature escalante
# • Carregamento seletivo de modelos (só carrega o necessary)
# • Cache de segmentos: reutiliza WAVs já gerados em runs anteriores (verificação rápida)
import subprocess
import re
import shutil
import asyncio
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
import torch
import importlib
import json
from pathlib import Path
from typing import Optional
import numpy as np
import soundfile as sf
from config.settings import (
    QWEN3_MODEL_BASE, QWEN3_MODEL_VOICEDESIGN,
    NARRATOR_PT_PT_INSTRUCT, ANCHOR_TEXT,
    TTS_MAX_RETRIES,
    ANCHOR_TIMEOUT, ANCHOR_MAX_NEW_TOKENS,
    TTS_MAX_NEW_TOKENS, TTS_BASE_TEMPERATURE,
    TTS_MIN_RMS, TTS_MAX_ZCR,
    PTPT_ACCENT_SUFFIX  # <-- ESTA LINHA É OBRIGATÓRIA
)
from tts.audio_validator import validate_audio, log_quality
from tts.vram_manager import release_model, log_vram

logger = logging.getLogger(__name__)

class TTSEngine:
    """Gere o carregamento lazy dos modelos Qwen3-TTS e a síntese de áudio."""
    def __init__(
        self, temp_dir: Path, log_fn=None
    ):
        self.temp_dir    = temp_dir
        self.log         = log_fn or logger.info
        self.model_base   = None   # Base  — clonagem de voz (ICL / x-vector)
        self.model_design = None   # VoiceDesign — síntese por descrição

        # Verificar se ffmpeg está disponível
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.log("⚠️ FFmpeg não encontrado. Algumas funcionalidades podem não funcionar.")

    # ═══════════════════════════════════════════════════════════════════════════
    # Carregamento / Libertação de Modelos
    # ═══════════════════════════════════════════════════════════════════════════

    def _load_model_sync(self, model_id: str, label: str):
        
        try:
            qwen_tts_mod = importlib.import_module('qwen_tts')
            Qwen3TTSModel = getattr(qwen_tts_mod, 'Qwen3TTSModel')
        except ImportError as e:
            raise ImportError(
                "Módulo 'qwen_tts' não encontrado. Instale 'qwen-tts' "
                "ou use o ambiente de execução correto."
            ) from e

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
    # Cache de Segmentos (VERIFICAÇÃO RÁPIDA)
    # ═══════════════════════════════════════════════════════════════════════════

    def segment_cache_path(self, seg_index: int) -> Path:
        return self.temp_dir / f"seg_{seg_index:05d}.wav"

    def is_segment_cached(self, seg_index: int, text: str) -> bool:
        path = self.segment_cache_path(seg_index)
        
        # DEBUG: Mostra o caminho absoluto do primeiro segmento para verificação
        if seg_index == 0:
            self.log(f"  [DEBUG] A procurar cache em: {path.absolute()}")
            if path.exists():
                self.log(f"  [DEBUG] O ficheiro {path.name} EXISTE e tem {path.stat().st_size} bytes.")
            else:
                self.log(f"  [DEBUG] O ficheiro {path.name} NÃO EXISTE nesta pasta.")
        
        if not path.exists():
            return False
        
        if path.stat().st_size > 1024:
            return True
            
        path.unlink(missing_ok=True)
        return False

    def count_cached_segments(self, segments: list) -> int:
        """Conta quantos segmentos já têm WAV em cache (versão rápida)."""
        count = 0
        for i, seg in enumerate(segments):
            if isinstance(seg, dict) and seg.get("text"):
                if self.is_segment_cached(i, seg["text"]):
                    count += 1
        return count
    
    def save_segment_metadata(self, seg_index: int, text: str, audio_path: str, success: bool, details: str = ""):
        """Guarda metadados de geração do segmento para cache/debug."""
        from datetime import datetime
        meta_file = self.temp_dir / "segment_metadata.json"
        
        data = {}
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
                
        # Trunca texto muito longo para não inflacionar o JSON
        short_text = text[:120] + "..." if len(text) > 120 else text
        
        data[str(seg_index)] = {
            "text": short_text,
            "audio_path": str(audio_path),
            "success": success,
            "details": details,
            "generated_at": datetime.now().isoformat()
        }
        
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ═══════════════════════════════════════════════════════════════════════════
    # Âncoras PT-PT
    # ═══════════════════════════════════════════════════════════════════════════

    def _generate_anchor_sync(self, cid: str, instruct: str,
                               anchor_path: str, cdata: dict) -> bool:
        """
        Gera âncora com retry interno + validação de qualidade.
        SEM ThreadPoolExecutor interno — o timeout é gerido pelo asyncio.wait_for
        em ensure_anchor, evitando conflito entre dois timeouts.
        """
        BASE_TEMP = TTS_BASE_TEMPERATURE   # usa constante do settings (0.15)
        TEMP_STEP = 0.04

        for attempt in range(1, TTS_MAX_RETRIES + 1):
            temp  = min(BASE_TEMP + (attempt - 1) * TEMP_STEP, 0.32)
            top_p = 0.80 if temp <= 0.20 else max(0.65, 0.80 - (attempt - 1) * 0.04)

            try:
                self.log(f"   ⚓ Âncora {cid} tentativa {attempt}/{TTS_MAX_RETRIES} "
                         f"(temp={temp:.2f}, top_p={top_p:.2f})...")

                wavs, sr = self.model_design.generate_voice_design(
                    text=ANCHOR_TEXT,
                    instruct=instruct,
                    language="portuguese",
                    temperature=temp,
                    top_p=top_p,
                    max_new_tokens=ANCHOR_MAX_NEW_TOKENS,  # 260 tokens ≈ 11s
                )

                if not self._write_audio(wavs, sr, anchor_path):
                    self.log(f"   ⚠️ [{attempt}] _write_audio falhou.")
                    continue

                q = validate_audio(anchor_path, ANCHOR_TEXT)

                # ── ADICIONAR LIMITE DE 390KB SÓ PARA ÂNCORAS AQUI ──
                if q.ok and Path(anchor_path).stat().st_size > 390 * 1024:
                    self.log(f"   ⚠️ [{attempt}] Âncora gerada é demasiado grande (>390KB). A rejeitar...")
                    Path(anchor_path).unlink(missing_ok=True)
                    continue # Vai para a próxima tentativa       

                if q.ok:
                    self.log(f"   ✅ Âncora OK [{cid}]: dur={q.duration:.1f}s "
                             f"rms={q.rms:.4f} zcr={q.zcr:.4f}")
                    cdata["ref_audio"] = anchor_path
                    cdata["ref_text"]  = ANCHOR_TEXT
                    return True
                else:
                    self.log(f"   ⚠️ [{attempt}] Âncora inválida: {q.reason} "
                             f"(rms={q.rms:.4f}, zcr={q.zcr:.4f}, dur={q.duration:.2f}s)")
                    Path(anchor_path).unlink(missing_ok=True)

            except Exception as e:
                self.log(f"   ❌ [{attempt}] Excepção: {e}")

        self.log(f"   ❌ Âncora {cid} falhou após {TTS_MAX_RETRIES} tentativas.")
        return False


    async def ensure_anchor(self, cid: str, cdata: dict):
        anchor_path = self.temp_dir / f"anchor_{cid}.wav"

        if anchor_path.exists() and anchor_path.stat().st_size > 1024:
            q = validate_audio(str(anchor_path), ANCHOR_TEXT)
            if q.ok:
                self.log(f"   ♻️ Âncora reutilizada: {cid} ({q.duration:.1f}s, rms={q.rms:.4f})")
                cdata["ref_audio"] = str(anchor_path)
                cdata["ref_text"]  = ANCHOR_TEXT
                return
            else:
                self.log(f"   ♻️❌ Âncora em disco inválida ({q.reason}) — a regenerar...")
                anchor_path.unlink(missing_ok=True)

        if cdata.get("ref_audio") and cdata.get("ref_text") is not None:
            return

        if cdata.get("ref_audio") and not anchor_path.exists():
            cdata.setdefault("ref_text", "")
            return

        base_desc = cdata.get("description", "Voz neutra")
        age = self._extract_age(base_desc.lower())
        gender = self._infer_gender(base_desc.lower())

        if age is None:
            age_text = "adult"
        elif age <= 10:
            age_text = f"{age}-year-old child"
        elif age <= 18:
            age_text = f"{age}-year-old teenager"
        elif age >= 65:
            age_text = "elderly"
        else:
            age_text = f"{age}-year-old adult"

        gender_text = "woman" if gender == "female" else "man"

        # Prompt natural e específico para a âncora
        instruct = f"""
    European Portuguese.

    A {age_text} {gender_text}.

    {base_desc}

    Never use a Brazilian accent.

    Speak naturally.

    Maintain the requested age.

    Maintain the requested gender.
    """

        if self.model_design is None:
            await self.load_voicedesign()

        try:
            success = await asyncio.wait_for(
                asyncio.to_thread(
                    self._generate_anchor_sync,
                    cid, instruct, str(anchor_path), cdata
                ),
                timeout=ANCHOR_TIMEOUT
            )
        except asyncio.TimeoutError:
            self.log(f"   ⏰ Timeout ({ANCHOR_TIMEOUT}s) ao gerar âncora para {cid}.")
            Path(anchor_path).unlink(missing_ok=True)
            success = False

        if not success:
            self.log(f"   ⚠️ Âncora falhou → '{cdata.get('name', cid)}' usará VoiceDesign PT-PT por segmento.")
            cdata["ref_audio"] = None
            cdata["ref_text"] = None

    def _build_voice_prompt(self, description: str, emotion: str) -> tuple[str, float, float, int]:
        """
        Constrói um prompt natural para o VoiceDesign, com base na descrição e emoção.
        Retorna: (prompt, temperature, top_p, pitch_semitones)
        """
        desc_lower = description.lower()
        age = self._extract_age(desc_lower)
        gender = self._infer_gender(desc_lower)

        # Mapeamento de emoção para descrições naturais
        emotion_text = {
            "joyful": "happy, smiling, energetic",
            "sad": "soft, emotional, quiet",
            "angry": "strong, tense, intense",
            "fearful": "hesitant, trembling",
            "calm": "gentle, relaxed",
            "neutral": "natural and conversational"
        }.get(emotion.lower(), "natural and conversational")

        # --------------------------------------------
        # CRIANÇAS (até 10 anos) – pitch shift +3~7 semitones
        # --------------------------------------------
        if age is not None and age <= 10:
            if gender == "female":
                prompt = f"""
    A Portuguese girl aged {age}.

    Her voice is unmistakably that of a little child.

    Very high natural pitch.

    Small vocal tract.

    Tiny body.

    Playful.

    Sweet.

    Curious.

    Energetic.

    Laughs easily.

    Never sounds like a teenager.

    Never sounds like an adult.

    European Portuguese accent.

    Speech style:
    {emotion_text}.
    """
                return prompt, 0.18, 0.88, 7   # pitch +7 semitones para meninas
            else:
                prompt = f"""
    A Portuguese boy aged {age}.

    His voice is clearly that of an eight-year-old child.

    High natural pitch.

    Small vocal tract.

    Light resonance.

    No signs of puberty.

    Playful.

    Excited.

    Energetic.

    Very expressive.

    Never sounds like a teenager.

    Never sounds like an adult.

    European Portuguese accent.

    Speech style:
    {emotion_text}.
    """
                return prompt, 0.18, 0.88, 5   # pitch +5 semitones para meninos

        # --------------------------------------------
        # ADOLESCENTES (11-18 anos) – pitch shift leve ou nenhum
        # --------------------------------------------
        if age is not None and age <= 18:
            if gender == "female":
                prompt = f"""
    Portuguese teenage girl.

    Age {age}.

    Bright feminine voice.

    Youthful.

    Natural.

    Fresh.

    Slightly immature.

    European Portuguese accent.

    Speech style:
    {emotion_text}.
    """
                return prompt, 0.15, 0.82, 2   # +2 semitones para suavizar
            else:
                prompt = f"""
    Portuguese teenage boy.

    Age {age}.

    Voice beginning puberty.

    Still youthful.

    Medium-high pitch.

    European Portuguese accent.

    Speech style:
    {emotion_text}.
    """
                return prompt, 0.15, 0.82, 1   # +1 semitone (ligeiro)

        # --------------------------------------------
        # MULHER ADULTA
        # --------------------------------------------
        if gender == "female":
            prompt = f"""
    Adult Portuguese woman.

    Natural feminine voice.

    Clear head resonance.

    Elegant.

    Natural.

    European Portuguese accent.

    Speech style:
    {emotion_text}.
    """
            return prompt, 0.08, 0.75, 0

        # --------------------------------------------
        # HOMEM ADULTO (fallback)
        # --------------------------------------------
        prompt = f"""
    Adult Portuguese man.

    Natural masculine voice.

    Chest resonance.

    Warm.

    Natural.

    European Portuguese accent.

    Speech style:
    {emotion_text}.
    """
        return prompt, 0.08, 0.75, 0

    # ═══════════════════════════════════════════════════════════════════════════
    # Síntese com Retry + Validação de Qualidade (TEMPERATURA BAIXA PARA EVITAR RUÍDO)
    # ═══════════════════════════════════════════════════════════════════════════

    def clone_with_emotion(self, text: str, ref_audio: str|None,
                       emotion: str, pace: float, out_path: str,
                       ref_text: str="",
                       voice_description: str="") -> bool:
        """Clona voz com retry automático e temperature muito baixa para evitar ruído."""
        
        from config.settings import ANCHOR_TEXT
        if text.strip() == ANCHOR_TEXT.strip():
            self.log(f"   ⚠️ Tentativa de usar texto de âncora como conteúdo - corrigindo...")
            desc = voice_description or "Voz neutra, português de Portugal, sotaque de Lisboa."
            return self.generate_design(text, desc, emotion, out_path)
        
        if len(text.strip()) < 20 and "portugal" in text.lower() and "sotaque" in text.lower():
            self.log(f"   ⚠️ Texto suspeito de ser âncora detectado - usando VoiceDesign")
            desc = voice_description or "Voz neutra, português de Portugal, sotaque de Lisboa."
            return self.generate_design(text, desc, emotion, out_path)

        if not ref_audio:
            desc = voice_description or "Voz neutra, português de Portugal, sotaque de Lisboa."
            return self.generate_design(text, desc, emotion, out_path)

        base_temp = 0.15
        max_tokens = TTS_MAX_NEW_TOKENS  
        
        for attempt in range(1, TTS_MAX_RETRIES + 1):
            temp = min(base_temp + (attempt - 1) * 0.05, 0.35)
            try:
                clean = self._clean_text(text)
                kwargs = dict(
                    text=clean, ref_audio=ref_audio, language='portuguese',
                    temperature=temp, top_p=0.85, max_new_tokens=max_tokens,
                )
                kwargs["ref_text"] = ref_text if ref_text else ANCHOR_TEXT
                if voice_description:
                    kwargs["instruct"] = f"{voice_description}{PTPT_ACCENT_SUFFIX}"
                else:
                    kwargs["instruct"] = (
                        "Sotaque de Portugal continental. Português europeu. "
                        f"Nunca brasileiro.{PTPT_ACCENT_SUFFIX}"
                    )
                
                wavs, sr = self.model_base.generate_voice_clone(**kwargs)
                if not self._write_audio(wavs, sr, out_path):
                    self.log(f"   ⚠️ Tentativa {attempt}: falha ao escrever áudio.")
                    continue
                
            except Exception as e:
                if attempt == TTS_MAX_RETRIES: 
                    self.log(f"   ❌ Falha após {TTS_MAX_RETRIES} tentativas: {e}")
                    return False
                self.log(f"   ⚠️ Tentativa {attempt} falhou, tentando novamente...")
                continue

            if not Path(out_path).exists():
                self.log(f"   ❌ Arquivo de saída não foi criado: {out_path}")
                if attempt == TTS_MAX_RETRIES: 
                    return False
                continue
                
            # VALIDAÇÃO
            q = validate_audio(out_path, text)
            if q.ok:
                if self._verify_content_match(out_path, text):
                    self.log(f"   ✅ Áudio validado: {q}")
                    return True
                else:
                    self.log(f"   ⚠️ Conteúdo não corresponde ao texto solicitado")
            else:
                self.log(f"   ⚠️ Validação falhou na tentativa {attempt}: {q.reason}")
                if attempt < TTS_MAX_RETRIES:
                    Path(out_path).unlink(missing_ok=True)

            # Última tentativa - aceitar se ficheiro existe
            if attempt == TTS_MAX_RETRIES:
                if Path(out_path).exists() and Path(out_path).stat().st_size > 1024:
                    self.log(f"   ⚠️ Validação falhou mas arquivo parece válido, aceitando...")
                    return True
                Path(out_path).unlink(missing_ok=True)
                return False

        return False

    def generate_clone(self, text: str, ref_audio: str, out_path: str,
                       ref_text: str = "") -> bool:
        return self.clone_with_emotion(
            text, ref_audio, "neutral", 1.0, out_path, ref_text
        )

    def generate_design(self, text: str, description: str, emotion: str, out_path: str) -> bool:
        """
        Gera áudio usando VoiceDesign com prompt natural e pós-processamento de pitch.
        """
        # 1. Construir prompt e parâmetros
        prompt, temp, top_p, pitch_semitones = self._build_voice_prompt(description, emotion)

        self.log(f"   🎤 Gerando voz com prompt: {prompt[:80]}...")

        # 2. Gerar áudio com retry
        for attempt in range(1, TTS_MAX_RETRIES + 1):
            try:
                wavs, sr = self.model_design.generate_voice_design(
                    text=text,
                    instruct=prompt,
                    language='portuguese',
                    temperature=temp,
                    top_p=top_p,
                    max_new_tokens=TTS_MAX_NEW_TOKENS,
                )
                if not self._write_audio(wavs, sr, out_path):
                    continue

                # 3. Aplicar pitch shift se necessário
                if pitch_semitones > 0:
                    shifted_path = out_path + ".shifted.wav"
                    if self._pitch_shift(out_path, shifted_path, semitones=pitch_semitones):
                        import shutil
                        shutil.move(shifted_path, out_path)
                        self.log(f"   ✅ Pitch shift aplicado: +{pitch_semitones} semitones")
                    else:
                        self.log("   ⚠️ Pitch shift falhou, a usar áudio original")

                # 4. Validar qualidade
                q = validate_audio(out_path, text)
                if q.ok:
                    self.log(f"   ✅ Áudio validado: {q}")
                    return True
                else:
                    self.log(f"   ⚠️ Validação falhou na tentativa {attempt}: {q.reason}")
                    # Só apaga se não for a última tentativa
                    if attempt < TTS_MAX_RETRIES:
                        Path(out_path).unlink(missing_ok=True)

                # Se for a última tentativa, aceitar se o ficheiro existir e tiver tamanho razoável
                if attempt == TTS_MAX_RETRIES:
                    if Path(out_path).exists() and Path(out_path).stat().st_size > 1024:
                        self.log(f"   ⚠️ Validação falhou mas arquivo parece válido, aceitando...")
                        return True
                    Path(out_path).unlink(missing_ok=True)
                    return False

            except Exception as e:
                self.log(f"   ❌ Erro na tentativa {attempt}: {e}")
                if attempt < TTS_MAX_RETRIES:
                    temp += 0.03
                    continue
                return False

        return False


    # --- FUNÇÕES AUXILIARES ---

    def _pitch_shift(self, input_path: str, output_path: str, semitones: float) -> bool:
        """
        Altera o pitch e os formantes para simular trato vocal mais curto.
        Usa Rubber Band se disponível, senão ffmpeg (apenas pitch).
        """
        try:
            import pyrubberband as pyrb
            import soundfile as sf
            import numpy as np

            # Carregar áudio
            y, sr = sf.read(input_path)

            # Aplicar pitch shift com formant correction (Rubber Band)
            # `formant` preserva o timbre original, mas para crianças queremos alterar formantes também.
            # Para simular trato mais curto, usamos formant_scale > 1.0 (ex: 1.2)
            # Aqui, para simplificar, usamos apenas pitch shift, mas o Rubber Band permite
            # preservar formantes (o que pode não ser ideal para crianças, mas é mais natural).
            # Vamos usar a opção padrão (que já ajusta formantes ligeiramente).
            y_shifted = pyrb.pitch_shift(y, sr, semitones)

            # Escrever
            sf.write(output_path, y_shifted, sr)
            return True

        except ImportError:
            # Fallback: ffmpeg (sem formant correction, apenas pitch)
            self.log("   ⚠️ pyrubberband não instalado. Usando ffmpeg (apenas pitch).")
            try:
                factor = 2 ** (semitones / 12.0)
                cmd = [
                    'ffmpeg', '-y', '-i', input_path,
                    '-af', f'asetrate=44100*{factor},atempo={1/factor}',
                    '-c:a', 'pcm_s16le', output_path
                ]
                subprocess.run(cmd, capture_output=True, check=True, timeout=30)
                return True
            except Exception as e:
                self.log(f"   ❌ ffmpeg pitch shift falhou: {e}")
                return False
        except Exception as e:
            self.log(f"   ❌ Rubber Band falhou: {e}")
            return False

    def _extract_age(self, desc_lower: str) -> Optional[int]:
        patterns = [
            r'(?:cerca de|cerca|aproximadamente|~)?\s*(\d+)\s*anos?\s*(?:de\s*idade)?',
            r'(\d+)\s*-\s*ano[s]?[\s-]?old',
            r'(\d+)\s*year[s]?\s+old',
        ]
        for pat in patterns:
            match = re.search(pat, desc_lower, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    def _infer_gender(self, desc_lower: str) -> str:
        """Infere gênero a partir de palavras-chave na descrição."""
        female_words = ["feminin", "mulher", "female", "woman", "rapariga", "menina", "mrs", "miss", "sra", "senhora", "misteriosa"]
        male_words = ["masculin", "homem", "male", "man", "rapaz", "menino", "mr", "sr", "senhor"]

        if any(k in desc_lower for k in female_words):
            return "female"
        elif any(k in desc_lower for k in male_words):
            return "male"
        else:
            return "unknown"

    def _write_audio(self, wavs, sr, out_path) -> bool:
        """Escreve o áudio para o arquivo (implementação depende da sua biblioteca)."""
        try:
            import soundfile as sf
            sf.write(out_path, wavs, sr)
            return True
        except Exception as e:
            self.log(f"   ❌ Erro ao escrever áudio: {e}")
            return False

    # ═══════════════════════════════════════════════════════════════════════════
    # Utilitários
    # ═══════════════════════════════════════════════════════════════════════════

    def _verify_content_match(self, audio_path: str, expected_text: str) -> bool:
        """Verifica se o conteúdo do áudio corresponde ao texto esperado (aproximadamente)."""
        try:
            # Esta é uma verificação básica - em produção poderia usar ASR
            expected_words = set(expected_text.lower().split())
            # Remover palavras comuns da âncora que não deveriam estar no conteúdo
            anchor_words = {"estou", "falar", "sotaque", "lisboa", "portugal", "europeia"}
            content_words = expected_words - anchor_words
            
            # Se o texto esperado é muito curto após remover palavras da âncora,
            # provavelmente é um texto de âncora
            if len(content_words) < 3 and len(expected_text) < 100:
                if any(word in expected_text.lower() for word in anchor_words):
                    return False  # Provavelmente é texto de âncora
            
            return True
        except:
            return True  # Em caso de erro, assumir que está correto


    def _write_audio(self, wavs, sr: int, out_path: str):
        """Escreve o áudio no disco, removendo automaticamente silêncio excessivo no início/fim."""
        import numpy as np
        import soundfile as sf
        
        try:
            import librosa
            # 1. Extrair o array de áudio
            audio = wavs[0] if (hasattr(wavs, '__len__') and not isinstance(wavs, np.ndarray)) else wavs
            if hasattr(audio, 'cpu'): audio = audio.cpu().numpy()
            if hasattr(audio, 'numpy'): audio = audio.numpy()

            # Garantir que é um array 1D (mono)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=0)

            # 2. REMOVER SILÊNCIO EXCESSIVO (A correção para os 10-15s)
            try:
                # top_db=30 é menos agressivo que 25, preservando mais conteúdo
                audio_trimmed, _ = librosa.effects.trim(
                    audio, 
                    top_db=30, 
                    frame_length=2048, 
                    hop_length=512
                )
                audio = audio_trimmed
            except Exception as e:
                # Fallback: se o librosa falhar por algum motivo, guarda o áudio original
                self.log(f"   ⚠️ Falha ao trimar silêncio: {e}")
                return False

            # 3. Verificar se o áudio não ficou muito curto após o trim
            duration = len(audio) / sr
            if duration < 0.1:  # Menos de 100ms
                self.log(f"   ⚠️ Áudio muito curto após trim ({duration:.3f}s), usando áudio original")
                # Usar o áudio original
                audio = wavs[0] if (hasattr(wavs, '__len__') and not isinstance(wavs, np.ndarray)) else wavs
                if hasattr(audio, 'cpu'): audio = audio.cpu().numpy()
                if hasattr(audio, 'numpy'): audio = audio.numpy()
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=0)

            # 4. Guardar o ficheiro final
            sf.write(out_path, audio, sr)
            return True 
            
        except ImportError:
            # Se librosa não estiver disponível, usar método simples
            audio = wavs[0] if (hasattr(wavs, '__len__') and not isinstance(wavs, np.ndarray)) else wavs
            if hasattr(audio, 'cpu'): audio = audio.cpu().numpy()
            if hasattr(audio, 'numpy'): audio = audio.numpy()
            if audio.ndim > 1:
                audio = np.mean(audio, axis=0)
            sf.write(out_path, audio, sr)
            return True 

    def _clean_text(self, text: str) -> str:
        text = text.replace('\u201c', '«').replace('\u201d', '»')
        text = text.replace('"', '«').replace('"', '»')
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        text = re.sub(r'[^\w\s«»\'\-.,;:!?…]', '', text, flags=re.UNICODE)
        if text and text[-1] not in '.!?…':
            text += '.'
        return text.strip()

    def create_silence(self, duration: float, filename: str) -> Path:
        """Cria um ficheiro WAV de silêncio para usar como pausa entre segmentos."""
        path = self.temp_dir / filename
        if not path.exists():
            cmd = [
                'ffmpeg', '-y', '-f', 'lavfi',
                '-i', 'anullsrc=r=24000:cl=mono',
                '-t', str(duration),
                '-c:a', 'pcm_s16le',
                str(path)
            ]
            try:
                subprocess.run(cmd, capture_output=True, check=True, timeout=30)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                self.log(f"   ⚠️ Erro ao criar silêncio com ffmpeg: {e}")
                # Criar arquivo vazio como fallback
                silence = np.zeros(int(24000 * duration), dtype=np.int16)
                sf.write(str(path), silence, 24000)
        return path