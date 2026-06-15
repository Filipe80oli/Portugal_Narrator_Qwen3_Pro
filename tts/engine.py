# ─── Motor TTS: Qwen3-TTS Base (clonagem) + VoiceDesign (síntese) ─────────────
# Melhorias v7.4:
# • Validação de qualidade de áudio (RMS, ZCR, silêncio, duração)
# • Retry automático com temperature escalante
# • Carregamento seletivo de modelos (só carrega o necessary)
# • Cache de segmentos: reutiliza WAVs já gerados em runs anteriores (verificação rápida)
import subprocess
import re
import asyncio
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
import torch
import numpy as np
import soundfile as sf
from config.settings import (
    QWEN3_MODEL_BASE, QWEN3_MODEL_VOICEDESIGN,
    NARRATOR_PT_PT_INSTRUCT, ANCHOR_TEXT,
    TTS_MAX_RETRIES, ANCHOR_MAX_RETRIES,    # ← adicionar
    ANCHOR_TIMEOUT, ANCHOR_MAX_NEW_TOKENS,
    TTS_MAX_NEW_TOKENS, TTS_BASE_TEMPERATURE,
    PTPT_ACCENT_SUFFIX
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

    # ═══════════════════════════════════════════════════════════════════════════
    # Âncoras PT-PT
    # ═══════════════════════════════════════════════════════════════════════════

    def _generate_anchor_sync(self, cid: str, instruct: str,
                               anchor_path: str, cdata: dict,
                               stop_event=None) -> bool:   # ← adicionar stop_event
        BASE_TEMP = TTS_BASE_TEMPERATURE
        TEMP_STEP = 0.03

        for attempt in range(1, ANCHOR_MAX_RETRIES + 1):
            if stop_event and stop_event.is_set():     # ← verificar no início de cada tentativa
                self.log("⛔ Geração de âncora cancelada.")
                return False

            temp  = min(BASE_TEMP + (attempt - 1) * TEMP_STEP, 0.36)
            top_p = max(0.65, 0.80 - (attempt - 1) * 0.02)

            try:
                self.log(f"   ⚓ Âncora {cid} tentativa {attempt}/{ANCHOR_MAX_RETRIES} "
                         f"(temp={temp:.2f}, top_p={top_p:.2f})...")

                wavs, sr = self.model_design.generate_voice_design(
                    text=ANCHOR_TEXT,
                    instruct=instruct,
                    language="portuguese",
                    temperature=temp,
                    top_p=top_p,
                    max_new_tokens=ANCHOR_MAX_NEW_TOKENS,
                )

                if not self._write_audio(wavs, sr, anchor_path):
                    self.log(f"   ⚠️ [{attempt}] _write_audio falhou.")
                    continue

                q = validate_audio(anchor_path, ANCHOR_TEXT)
                if q.ok:
                    self.log(f"   ✅ Âncora OK [{cid}]: dur={q.duration:.1f}s "
                             f"rms={q.rms:.4f} zcr={q.zcr:.4f}")
                    cdata["ref_audio"] = anchor_path
                    return True

                self.log(f"   ⚠️ [{attempt}] Âncora inválida: {q.reason} "
                         f"(rms={q.rms:.4f}, zcr={q.zcr:.4f}, dur={q.duration:.2f}s)")
                Path(anchor_path).unlink(missing_ok=True)

            except Exception as e:
                self.log(f"   ❌ [{attempt}] Excepção: {e}")

        self.log(f"   ❌ Âncora {cid} falhou após {ANCHOR_MAX_RETRIES} tentativas.")
        return False


    async def ensure_anchor(self, cid: str, cdata: dict):
        """
        Cria âncora de voz PT-PT (VoiceDesign) para personagens sem .wav externo.
        Primeiro verifica se já existe no disco, reutilizando se possível.
        """
        anchor_path = self.temp_dir / f"anchor_{cid}.wav"

        # ── PRIORIDADE 1: Verificar se já existe no disco (reutilizar) ────────
        # Substituir linhas 183-188:
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

        # ── PRIORIDADE 2: Já tem áudio e texto de referência → nada a fazer ───
        if cdata.get("ref_audio") and cdata.get("ref_text") is not None:
            return

        # ── PRIORIDADE 3: .wav externo fornecido pelo utilizador ───────────────
        if cdata.get("ref_audio") and not anchor_path.exists():
            cdata.setdefault("ref_text", "")
            return

        # ── PRIORIDADE 4: Gerar nova âncora com VoiceDesign PT-PT ────────────────────
        base_desc = cdata.get("description", "Voz neutra")
        
        # FORÇAR SEMPRE SOTAQUE PORTUGUÊS EUROPEU
        if cid == "narrator":
            instruct = NARRATOR_PT_PT_INSTRUCT
        else:
            # ANTES: instruct = f"Voz portuguesa de Portugal. {base_desc}. Sotaque europeu estrito."
            # DEPOIS: mesma força que o narrador, adaptada ao género/descrição
            instruct = (
                f"{base_desc}."
                f" Sotaque de Portugal continental, português europeu."
                f" Vogais fechadas, ritmo europeu. Nunca brasileiro."
                f" Dicção clara de Lisboa ou Porto."
            )

        # VoiceDesign precisa de estar carregado
        if self.model_design is None:
            await self.load_voicedesign()

        try:
            success = await asyncio.wait_for(
                asyncio.to_thread(
                    self._generate_anchor_sync,
                    cid, instruct, str(anchor_path), cdata,
                    self._stop_event if hasattr(self, '_stop_event') else None  # ← adicionar
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

    # ═══════════════════════════════════════════════════════════════════════════
    # Síntese com Retry + Validação de Qualidade (TEMPERATURA BAIXA PARA EVITAR RUÍDO)
    # ═══════════════════════════════════════════════════════════════════════════

    def clone_with_emotion(self, text: str, ref_audio: str|None,
                        emotion: str, pace: float, out_path: str,
                        ref_text: str="",
                        voice_description: str="") -> bool:
        """Clona voz com retry automático e temperature muito baixa para evitar ruído."""
        
        # VALIDAÇÃO CRÍTICA: Evitar usar texto da âncora como conteúdo
        from config.settings import ANCHOR_TEXT
        if text.strip() == ANCHOR_TEXT.strip():
            self.log(f"   ⚠️ Tentativa de usar texto de âncora como conteúdo - corrigindo...")
            # Se for exatamente o texto da âncora, usar VoiceDesign direto
            desc = voice_description or "Voz neutra, português de Portugal, sotaque de Lisboa."
            return self.generate_design(text, desc, emotion, out_path)
        
        # Evitar textos muito curtos que possam ser de âncoras
        if len(text.strip()) < 20 and "portugal" in text.lower() and "sotaque" in text.lower():
            self.log(f"   ⚠️ Texto suspeito de ser âncora detectado - usando VoiceDesign")
            desc = voice_description or "Voz neutra, português de Portugal, sotaque de Lisboa."
            return self.generate_design(text, desc, emotion, out_path)

        if not ref_audio:
            desc = voice_description or "Voz neutra, português de Portugal, sotaque de Lisboa."
            return self.generate_design(text, desc, emotion, out_path)

        # Temperature muito mais baixa + tokens limitados para evitar ruído
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

                # ref_text é a transcrição do áudio de referência — só passar se existir
                # NUNCA passar ANCHOR_TEXT: o modelo pode gerá-lo como fala no segmento
                if ref_text and ref_text.strip():
                    kwargs["ref_text"] = ref_text
                else:
                    kwargs["x_vector_only_mode"] = True

                # reforço PT-PT
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

            # Verificação adicional: garantir que o arquivo foi criado
            if not Path(out_path).exists():
                self.log(f"   ❌ Arquivo de saída não foi criado: {out_path}")
                if attempt == TTS_MAX_RETRIES: return False
                continue
                
            # VALIDAÇÃO MAIS RIGOROSA
            q = validate_audio(out_path, text)
            if q.ok:
                # Verificação extra: conteúdo realmente corresponde ao texto solicitado?
                if self._verify_content_match(out_path, text):
                    self.log(f"   ✅ Áudio validado: {q}")
                    return True
                else:
                    self.log(f"   ⚠️ Conteúdo não corresponde ao texto solicitado")
            else:
                self.log(f"   ⚠️ Validação falhou na tentativa {attempt}: {q.reason}")
                # Remover arquivo ruim
                Path(out_path).unlink(missing_ok=True)
                
            # Se for a última tentativa, mesmo com falha na validação, verificar se o arquivo existe
            if attempt == TTS_MAX_RETRIES:
                # NUNCA aceitar sem validação — áudio com prompt de âncora no início
                # passa nos 1024 bytes mas contamina o segmento
                self.log(f"   ❌ Todas as tentativas falharam na validação.")
                Path(out_path).unlink(missing_ok=True)
                return False

        return False

    def generate_clone(self, text: str, ref_audio: str, out_path: str,
                       ref_text: str = "") -> bool:
        return self.clone_with_emotion(
            text, ref_audio, "neutral", 1.0, out_path, ref_text
        )

    def generate_design(self, text: str, description: str, emotion: str, out_path: str) -> bool:

        is_narrator = "narrator" in description.lower() or "narrador" in description.lower()
        gender_fix = "Voz masculina, homem de Portugal. " if is_narrator else ""

        full_instruct = (
            f"{description}. {gender_fix}"
            "Sotaque de Portugal. Português Europeu. "
            f"Voz clara, sem ruído. Emoção: {emotion}."
            f"{PTPT_ACCENT_SUFFIX}"   # ← adicionar esta linha
        )

        # Temperatura base baixa para estabilidade; sobe ligeiramente a cada falha
        # para dar ao modelo margem de variação sem explodir para ruído
        BASE_TEMP = 0.15
        TEMP_STEP = 0.04
        MAX_TEMP  = 0.35  # nunca subir acima disto no VoiceDesign

        for attempt in range(1, TTS_MAX_RETRIES + 1):
            temp = min(BASE_TEMP + (attempt - 1) * TEMP_STEP, MAX_TEMP)
            # top_p desce ligeiramente com temp alta para compensar
            top_p = 0.8 if temp <= 0.2 else max(0.65, 0.8 - (attempt - 1) * 0.04)

            try:
                self.log(f"   🎙️ VoiceDesign tentativa {attempt}/{TTS_MAX_RETRIES} "
                        f"(temp={temp:.2f}, top_p={top_p:.2f})...")

                wavs, sr = self.model_design.generate_voice_design(
                    text=text,
                    instruct=full_instruct,
                    language='portuguese',
                    temperature=temp,
                    top_p=top_p,
                    max_new_tokens=TTS_MAX_NEW_TOKENS,
                )

                if not self._write_audio(wavs, sr, out_path):
                    self.log(f"   ⚠️ [{attempt}] _write_audio falhou (ruído ou write error)")
                    continue

                q = validate_audio(out_path, text)
                if q.ok:
                    if attempt > 1:
                        self.log(f"   ✅ VoiceDesign OK na tentativa {attempt}")
                    return True
                else:
                    self.log(f"   ⚠️ [{attempt}] Validação falhou: {q.reason} "
                            f"(rms={q.rms:.4f}, zcr={q.zcr:.4f})")

            except Exception as e:
                self.log(f"   ⚠️ [{attempt}] Excepção VoiceDesign: {e}")

        self.log(f"   ❌ VoiceDesign esgotou {TTS_MAX_RETRIES} tentativas para: {text[:40]}...")
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
            # Normalizar aspas
            text = text.replace('\u201c', '«').replace('\u201d', '»')
            text = text.replace('"', '«').replace('"', '»')
            text = text.replace('\u2018', "'").replace('\u2019', "'")

            # Remover travessão de diálogo no início — o modelo não precisa dele
            # mas preservar o conteúdo: "- Sim." → "Sim."
            text = re.sub(r'^[\-–—]\s*', '', text.strip())

            # Remover caracteres especiais mantendo pontuação PT
            text = re.sub(r'[^\w\s«»\'\-.,;:!?…]', '', text, flags=re.UNICODE)

            # Garantir pontuação final — crítico para textos curtos
            text = text.strip()
            if text and text[-1] not in '.!?…':
                text += '.'

            return text

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
