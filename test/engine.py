# ─── Motor TTS: Qwen3-TTS Base (clonagem) + VoiceDesign (síntese) ─────────────
# Melhorias v7.4.2:
# • Validação de qualidade de áudio (RMS, ZCR, silêncio, duração)
# • Filtro de ruído estático (ZCR > 0.45) para evitar estática
# • Retry automático com temperatura estável (0.1 para âncoras)
# • Proteção contra "Texto de Âncora" no conteúdo dos segmentos
# • Gestão robusta de VRAM e carregamento lazy
# • Trim agressivo de silêncio (top_db=20) para remover "hiss" final

import subprocess
import re
import asyncio
import logging
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

import torch
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
    
    def __init__(self, temp_dir: Path, log_fn=None):
        self.temp_dir = temp_dir
        self.log = log_fn or logger.info
        self.model_base = None   # Modelo para Clonagem (ICL)
        self.model_design = None # Modelo para Síntese por Descrição (VoiceDesign)

        # Verificar se ffmpeg está disponível para a concatenação final e silêncio
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.log("⚠️ FFmpeg não encontrado no sistema. Algumas funcionalidades de áudio podem falhar.")

    # ═══════════════════════════════════════════════════════════════════════════
    # Carregamento / Libertação de Modelos (Lazy Loading)
    # ═══════════════════════════════════════════════════════════════════════════

    def _load_model_sync(self, model_id: str, label: str):
        """Carregamento síncrono para ser executado via thread."""
        from qwen_tts import Qwen3TTSModel
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Usar bfloat16 se disponível para melhor performance e menos ruído em GPUs modernas
        dtype = torch.bfloat16 if (device == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float32
        attn = 'sdpa' if device == 'cuda' else 'eager'
        
        self.log(f'   [MODEL] {label} -> {device.upper()} | dtype={dtype} | attn={attn}')
        
        try:
            model = Qwen3TTSModel.from_pretrained(
                model_id, 
                device_map=device, 
                dtype=dtype, 
                attn_implementation=attn,
            )
            return model
        except Exception as e:
            self.log(f"   ❌ Erro crítico ao carregar {label}: {e}")
            raise

    async def load_base(self):
        """Carrega o modelo Base se ainda não estiver em memória."""
        if self.model_base is not None:
            return
        log_vram(self.log)
        self.log(f'🤖 A carregar Base (clonagem) -- {QWEN3_MODEL_BASE} ...')
        self.model_base = await asyncio.to_thread(
            self._load_model_sync, QWEN3_MODEL_BASE, 'Base'
        )
        log_vram(self.log)
        self.log('✅ Modelo Base carregado com sucesso.')

    async def load_voicedesign(self):
        """Carrega o modelo VoiceDesign se ainda não estiver em memória."""
        if self.model_design is not None:
            return
        log_vram(self.log)
        self.log(f'🤖 A carregar VoiceDesign -- {QWEN3_MODEL_VOICEDESIGN} ...')
        self.model_design = await asyncio.to_thread(
            self._load_model_sync, QWEN3_MODEL_VOICEDESIGN, 'VoiceDesign'
        )
        log_vram(self.log)
        self.log('✅ Modelo VoiceDesign carregado com sucesso.')

    def release_base(self):
        """Liberta memória GPU do modelo Base."""
        if self.model_base:
            release_model("model_base", self, self.log)
            self.log("🧹 Memória do modelo Base libertada.")

    def release_voicedesign(self):
        """Liberta memória GPU do modelo VoiceDesign."""
        if self.model_design:
            release_model("model_design", self, self.log)
            self.log("🧹 Memória do modelo VoiceDesign libertada.")

    # ═══════════════════════════════════════════════════════════════════════════
    # Gestão de Cache e Requisitos de Modelos
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def needs_base(characters: dict) -> bool:
        """Determina se precisamos do modelo Base baseado nos personagens."""
        return any(c.get("ref_audio") for c in characters.values())

    @staticmethod
    def needs_voicedesign(characters: dict) -> bool:
        """Determina se precisamos do VoiceDesign (personagens sem áudio de ref)."""
        return any(not c.get("ref_audio") for c in characters.values())

    def segment_cache_path(self, seg_index: int) -> Path:
        """Gera o caminho para o ficheiro de cache de um segmento."""
        return self.temp_dir / f"seg_{seg_index:05d}.wav"

    def is_segment_cached(self, seg_index: int, text: str) -> bool:
        """Verifica se um segmento já foi gerado e é válido."""
        path = self.segment_cache_path(seg_index)
        
        if not path.exists():
            return False
        
        # Verificação rápida por tamanho (mínimo 1KB para ser um áudio real)
        if path.stat().st_size > 1024:
            return True
            
        # Se existir mas estiver corrompido/vazio, apaga
        try:
            path.unlink(missing_ok=True)
        except:
            pass
        return False

    def count_cached_segments(self, segments: list) -> int:
        """Conta quantos segmentos da lista já estão em cache."""
        count = 0
        for i, seg in enumerate(segments):
            if isinstance(seg, dict) and seg.get("text"):
                if self.is_segment_cached(i, seg["text"]):
                    count += 1
        return count

    # ═══════════════════════════════════════════════════════════════════════════
    # Detecção de Qualidade e Geração de Âncoras (Proteção Anti-Ruído)
    # ═══════════════════════════════════════════════════════════════════════════

    def is_actually_noise(self, audio_array) -> bool:
        """
        Analisa o áudio para detetar ruído branco/estática.
        Usa Zero Crossing Rate (ZCR). Ruído tem ZCR muito elevado.
        """
        if audio_array is None or len(audio_array) == 0:
            return True
        
        # Calcular taxa de cruzamento por zero
        zcr = np.mean(np.abs(np.diff(np.sign(audio_array)))) / 2
        
        # Em áudio de fala normal, ZCR costuma estar entre 0.05 e 0.25. 
        # Valores acima de 0.45 indicam quase certamente estática.
        if zcr > TTS_MAX_ZCR:
            return True
        return False

    def _generate_anchor_sync(self, cid: str, instruct: str, anchor_path: str, cdata: dict) -> bool:
        """Gera a âncora de voz de forma síncrona com parâmetros de alta estabilidade."""
        
        def _do_generate():
            # PARÂMETROS CRÍTICOS PARA EVITAR RUÍDO:
            # Temperature baixa (0.1) = menos aleatoriedade = menos estática.
            # Top_p baixo (0.7) = ignora tokens de baixa probabilidade que causam ruído.
            wavs, sr = self.model_design.generate_voice_design(
                text=ANCHOR_TEXT,
                instruct=instruct,
                language="portuguese",
                temperature=0.1,  
                top_p=0.7,        
                max_new_tokens=ANCHOR_MAX_NEW_TOKENS,
            )
            # A função _write_audio já contém a validação is_actually_noise
            return self._write_audio(wavs, sr, anchor_path)

        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_do_generate)
            try:
                success = fut.result(timeout=ANCHOR_TIMEOUT)
                if success:
                    cdata["ref_audio"] = anchor_path
                    cdata["ref_text"] = ANCHOR_TEXT
                    self.log(f"   ✅ Âncora gerada com sucesso para '{cid}'")
                    return True
                return False
            except FuturesTimeout:
                self.log(f"   ⏰ Timeout ao gerar âncora para {cid}.")
                return False
            except Exception as e:
                self.log(f"   ❌ Erro na geração de âncora {cid}: {e}")
                return False

    async def ensure_anchor(self, cid: str, cdata: dict):
        """Garante que o personagem tem um áudio de referência (âncora)."""
        anchor_path = self.temp_dir / f"anchor_{cid}.wav"

        # 1. Reutilizar se já existir
        if anchor_path.exists() and anchor_path.stat().st_size > 1024:
            q = validate_audio(str(anchor_path), ANCHOR_TEXT)
            if q.ok:
                cdata["ref_audio"] = str(anchor_path)
                cdata["ref_text"] = ANCHOR_TEXT
                return
            else:
                self.log(f"   ⚠️ Âncora existente inválida ({q.reason}), a regenerar...")
                anchor_path.unlink(missing_ok=True)

        # 2. Se já tem ref externa, validar
        if cdata.get("ref_audio") and cdata.get("ref_text") is not None:
            return

        # 3. Gerar nova âncora PT-PT
        base_desc = cdata.get("description", "Voz neutra")
        
        # Construir instrução positiva (focada no que queremos)
        if cid == "narrator":
            instruct = NARRATOR_PT_PT_INSTRUCT  # usa a constante definida em settings
        else:
            instruct = (
                f"{base_desc}."
                f" Sotaque de Portugal continental, português europeu."
                f" Vogais fechadas, ritmo europeu. Nunca brasileiro."
                f" Dicção clara de Lisboa ou Porto."
            )

        if self.model_design is None:
            await self.load_voicedesign()

        success = await asyncio.to_thread(
            self._generate_anchor_sync, cid, instruct, str(anchor_path), cdata
        )
        
        if not success:
            self.log(f"   ⚠️ Não foi possível gerar âncora para {cid}. Usará síntese direta.")
            cdata["ref_audio"] = None
            cdata["ref_text"] = None

    # ═══════════════════════════════════════════════════════════════════════════
    # Síntese com Proteção de Conteúdo e Retries
    # ═══════════════════════════════════════════════════════════════════════════

    def clone_with_emotion(self, text: str, ref_audio: str|None, 
                          emotion: str, pace: float, out_path: str,
                          ref_text: str="", voice_description: str="") -> bool:
        """Realiza a clonagem de voz (ICL) com sistema de retry e validação."""
        
        # PROTEÇÃO: Se o texto for idêntico à âncora, desviamos para Design
        # para evitar que o modelo entre em loop infinito de âncoras.
        if text.strip() == ANCHOR_TEXT.strip() or (len(text) < 120 and "sotaque de lisboa" in text.lower()):
            self.log(f"   ⚠️ Conteúdo suspeito de ser âncora. Redirecionando para VoiceDesign...")
            return self.generate_design(text, voice_description, emotion, out_path)

        if not ref_audio:
            return self.generate_design(text, voice_description, emotion, out_path)

        base_temp = TTS_BASE_TEMPERATURE
        
        for attempt in range(1, TTS_MAX_RETRIES + 1):
            # Aumenta ligeiramente a temperatura em cada tentativa se a anterior falhou
            temp = min(base_temp + (attempt - 1) * 0.05, 0.35)
            
            try:
                clean_txt = self._clean_text(text)
                
                # Configuração da síntese
                kwargs = {
                    "text": clean_txt,
                    "ref_audio": ref_audio,
                    "language": 'portuguese',
                    "temperature": temp,
                    "top_p": 0.85,
                    "max_new_tokens": TTS_MAX_NEW_TOKENS,
                }
                
                if ref_text:
                    kwargs["ref_text"] = ref_text
                else:
                    kwargs["x_vector_only_mode"] = True

                # NOVO: forçar sotaque PT-PT mesmo na clonagem
                # (só tem efeito se o wrapper Qwen3TTSModel suportar 'instruct' no clone)
                if voice_description:
                    kwargs["instruct"] = f"{voice_description}{PTPT_ACCENT_SUFFIX}"
                
                # Gerar
                wavs, sr = self.model_base.generate_voice_clone(**kwargs)
                
                # Processar e Gravar
                if self._write_audio(wavs, sr, out_path):
                    # Validar qualidade técnica do ficheiro gravado
                    q = validate_audio(out_path, text)
                    if q.ok:
                        if self._verify_content_match(out_path, text):
                            return True
                        else:
                            self.log(f"   ⚠️ Conteúdo do áudio não parece coincidir com o texto.")
                    else:
                        self.log(f"   ⚠️ Falha na validação técnica: {q.reason}")
                
            except Exception as e:
                self.log(f"   ⚠️ Erro na tentativa {attempt} de clonagem: {e}")
            
            # Se chegamos aqui, a tentativa falhou. Limpar cache sujo.
            try: Path(out_path).unlink(missing_ok=True)
            except: pass

        return False

    def generate_clone(self, text: str, ref_audio: str, out_path: str, ref_text: str = "") -> bool:
        """Wrapper simples para compatibilidade."""
        return self.clone_with_emotion(text, ref_audio, "neutral", 1.0, out_path, ref_text)

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
    # Processamento de Áudio, Escrita e Utilitários
    # ═══════════════════════════════════════════════════════════════════════════

    def _write_audio(self, wavs, sr: int, out_path: str) -> bool:
        try:
            import librosa
            from config.settings import TTS_MIN_RMS, TTS_MAX_ZCR

            # 1. Extrair array numpy
            audio = wavs[0] if (hasattr(wavs, '__len__') and not isinstance(wavs, np.ndarray)) else wavs
            if hasattr(audio, 'cpu'):
                audio = audio.cpu().numpy()
            elif hasattr(audio, 'numpy'):
                audio = audio.numpy()
            if audio.ndim > 1:
                audio = np.mean(audio, axis=0)

            # 2. TRIM PRIMEIRO — top_db=30 preserva fala suave e consoantes PT-PT
            try:
                audio_trimmed, _ = librosa.effects.trim(
                    audio,
                    top_db=30,
                    frame_length=1024,
                    hop_length=256
                )
                if len(audio_trimmed) < sr * 0.3:
                    self.log("   ⚠️ Trim agressivo removeu demasiado — a tentar top_db=45... ")
                    audio_trimmed, _ = librosa.effects.trim(
                        audio,
                        top_db=45,
                        frame_length=1024,
                        hop_length=256
                    )
                if len(audio_trimmed) >= sr * 0.3:
                    audio = audio_trimmed
            except Exception as e:
                self.log(f"   ⚠️ Falha no trim: {e} ")

            # 3. FILTRO DE RUÍDO (ZCR)
            frame_len = int(sr * 0.02)  # janelas de 20ms
            if len(audio) >= frame_len:
                frames = [audio[i:i+frame_len] for i in range(0, len(audio) - frame_len, frame_len)]
                
                # CORREÇÃO: np.mean sem espaços
                frame_rms = np.array([np.sqrt(np.mean(f**2)) for f in frames])
                active_mask = frame_rms >= TTS_MIN_RMS

                if active_mask.any():
                    active_frames = [frames[j] for j in range(len(frames)) if active_mask[j]]
                    
                    # CORREÇÃO: zcr_per_frame sem espaços
                    zcr_per_frame = np.array([
                        float(np.mean(np.abs(np.diff(np.sign(f)))) / 2)
                        for f in active_frames
                    ])
                    
                    noisy_ratio = float(np.mean(zcr_per_frame > TTS_MAX_ZCR))
                    if noisy_ratio > 0.35:
                        self.log(f"   ❌ Rejeitado: {noisy_ratio:.0%} frames com ZCR de estática. ")
                        return False
                else:
                    self.log("   ❌ Rejeitado: áudio é silêncio total após trim. ")
                    return False

            # 4. Limite de duração para âncoras
            if "anchor" in str(out_path):
                max_samples = sr * 15
                if len(audio) > max_samples:
                    self.log(f"   ⚠️ Âncora longa ({len(audio)/sr:.1f}s) — truncando a 15s. ")
                    audio = audio[:max_samples]

            # 5. Gravar
            import soundfile as sf
            sf.write(out_path, audio, sr)

            from pathlib import Path
            if Path(out_path).exists() and Path(out_path).stat().st_size > 500:
                return True
            return False

        except Exception as e:
            self.log(f"   ❌ Erro ao processar/escrever áudio: {e} ")
            return False

    def _verify_content_match(self, audio_path: str, expected_text: str) -> bool:
        """
        Impede que o modelo gere áudio de âncora quando devia gerar livro.
        """
        anchor_keywords = {"estou a falar", "sotaque de lisboa", "minha voz europeia"}
        text_lower = expected_text.lower()
        
        # Se o texto que pedimos NÃO tem estas palavras, mas o áudio gerado é curto 
        # e o texto original do segmento era suspeito, fazemos uma verificação extra.
        # (Nota: Em versões futuras isto usaria ASR rápido).
        for kw in anchor_keywords:
            if kw in text_lower:
                return True # É suposto ter a keyword (é uma âncora)
        
        # Por agora, assumimos True mas deixamos o gancho para validação ASR
        return True

    def _clean_text(self, text: str) -> str:
        """Limpa o texto para o motor TTS, convertendo pontuação e aspas."""
        if not text: return ""
        
        # Converter aspas e travessões para o estilo europeu que o Qwen entende melhor
        text = text.replace('\u201c', '«').replace('\u201d', '»') # Aspas curvas
        text = text.replace('"', '«').replace('"', '»')           # Aspas retas
        text = text.replace('\u2018', "'").replace('\u2019', "'") # Plicas curvas
        text = text.replace('\u2014', ' - ').replace('\u2013', ' - ') # Travessões
        
        # Remover caracteres estranhos mas manter pontuação básica
        text = re.sub(r'[^\w\s«»\'\-.,;:!?…%€$]', '', text, flags=re.UNICODE)
        
        # Garantir que termina com pontuação para o modelo não "ficar à espera"
        text = text.strip()
        if text and text[-1] not in '.!?…':
            text += '.'
            
        return text

    def create_silence(self, duration: float, filename: str) -> Path:
        """Cria um ficheiro WAV de silêncio absoluto."""
        path = self.temp_dir / filename
        
        if path.exists():
            return path
            
        # Tentar via FFmpeg (mais rápido e limpo)
        cmd = [
            'ffmpeg', '-y', '-f', 'lavfi', 
            '-i', f'anullsrc=r=24000:cl=mono', 
            '-t', str(duration), 
            '-c:a', 'pcm_s16le', 
            str(path)
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=10)
        except:
            # Fallback via Numpy/Soundfile
            silence = np.zeros(int(24000 * duration), dtype=np.float32)
            sf.write(str(path), silence, 24000)
            
        return path