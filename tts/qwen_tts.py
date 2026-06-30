# qwen_tts.py
# Wrapper para o modelo Qwen3-TTS com suporte a VoiceDesign e Voice Clone
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Qwen3TTSModel:
    """Wrapper para o modelo Qwen3-TTS."""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()  # Modo de avaliação
        
    @classmethod
    def from_pretrained(cls, model_id, device_map='cuda', dtype=torch.float16, attn_implementation='sdpa'):
        """Carrega o modelo Qwen3-TTS do HuggingFace."""
        
        # Carregar tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        # Carregar modelo com as configurações especificadas
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=True
        )
        
        return cls(model, tokenizer, device_map)

    def generate_voice_design(self, text, instruct, language, temperature, top_p, max_new_tokens):
        """
        Gera áudio a partir de texto usando VoiceDesign.
        
        Args:
            text: Texto a sintetizar
            instruct: Descrição da voz
            language: Idioma (ex: "portuguese")
            temperature: Temperatura para sampling
            top_p: Top-p sampling
            max_new_tokens: Máximo de tokens a gerar
            
        Returns:
            tuple: (waveform, sample_rate)
        """
        # Construir o prompt no formato Qwen-TTS
        prompt = self._build_voicedesign_prompt(text, instruct, language)
        
        # Tokenizar
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Gerar áudio
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decodificar tokens em waveform
        waveform = self._decode_audio_tokens(outputs)
        
        # Qwen3-TTS usa 24000 Hz por padrão
        sample_rate = 24000
        
        return waveform, sample_rate

    def generate_voice_clone(self, text, ref_audio, language, temperature, top_p, 
                            max_new_tokens, ref_text="", instruct="", 
                            x_vector_only_mode=False):
        """
        Gera áudio clonando uma voz de referência (ICL - In-Context Learning).
        """
        import torchaudio
        
        # Carregar áudio de referência
        if isinstance(ref_audio, str):
            wav, sr = torchaudio.load(ref_audio)
        else:
            wav = ref_audio
            sr = 24000
        
        # Converter para mono se necessário
        if wav.dim() > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        # Resample se necessário (Qwen3-TTS espera 24kHz)
        if sr != 24000:
            resampler = torchaudio.transforms.Resample(sr, 24000)
            wav = resampler(wav)
        
        # Construir prompt de clonagem
        prompt = self._build_clone_prompt(
            text, 
            ref_text if not x_vector_only_mode else "",
            instruct,
            language
        )
        
        # Tokenizar
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Extrair e adicionar x-vector apenas se não estiver em modo apenas x-vector
        # e se o modelo suportar injeção de áudio de referência
        if not x_vector_only_mode:
            x_vector = self._extract_x_vector(wav)
            if x_vector is not None:
                inputs['ref_audio'] = x_vector.to(self.device)
        
        # Gerar áudio
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decodificar tokens em waveform
        waveform = self._decode_audio_tokens(outputs)
        sample_rate = 24000
        
        return waveform, sample_rate

    def _build_voicedesign_prompt(self, text, instruct, language):
        """Constrói o prompt para VoiceDesign."""
        prompt = (
            f"<|im_start|>system\n"
            f"Generate speech for the following text with the specified voice characteristics.\n"
            f"Language: {language}\n"
            f"Voice: {instruct}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return prompt

    def _build_clone_prompt(self, text, ref_text, instruct, language):
        """Constrói o prompt para clonagem de voz."""
        if ref_text:
            prompt = (
                f"<|im_start|>system\n"
                f"Clone the voice from the reference audio and generate speech.\n"
                f"Language: {language}<|im_end|>\n"
                f"<|im_start|>user\n"
                f"Reference text: {ref_text}\n"
                f"Target text: {text}\n"
                f"Voice characteristics: {instruct}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        else:
            prompt = (
                f"<|im_start|>system\n"
                f"Clone the voice from reference audio embedding.\n"
                f"Language: {language}<|im_end|>\n"
                f"<|im_start|>user\n"
                f"Target text: {text}\n"
                f"Voice characteristics: {instruct}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        return prompt

    def _extract_x_vector(self, wav):
        """
        Extrai x-vector (embedding de voz) do áudio de referência.
        Implementação simplificada usando MFCCs.
        """
        try:
            import librosa
            # Converter para numpy
            wav_np = wav.cpu().numpy().flatten()
            
            # Extrair MFCCs
            mfccs = librosa.feature.mfcc(y=wav_np, sr=24000, n_mfcc=40)
            # Média das MFCCs ao longo do tempo
            x_vector = torch.tensor(mfccs.mean(axis=1), dtype=torch.float32)
            # Expandir para dimensão esperada
            x_vector = x_vector.unsqueeze(0)  # (1, 40)
        except ImportError:
            # Fallback: usar estatísticas simples do áudio
            wav_np = wav.cpu().numpy().flatten()
            x_vector = torch.tensor([
                wav_np.mean(),
                wav_np.std(),
                wav_np.max(),
                wav_np.min(),
            ], dtype=torch.float32).unsqueeze(0)
        
        return x_vector

    def _decode_audio_tokens(self, outputs):
        """
        Decodifica tokens de saída em waveform de áudio.
        
        NOTA: Esta implementação depende da arquitetura específica do Qwen3-TTS.
        Consulta a documentação oficial para a implementação correta.
        """
        # Extrair tokens de áudio (últimos tokens da sequência)
        audio_tokens = outputs[:, -1000:]  # Ajustar conforme necessário
        
        # Verificar se o modelo tem um audio_head
        if hasattr(self.model, 'audio_head'):
            # Usar audio head para converter tokens em waveform
            waveform = self.model.audio_head(audio_tokens)
            return waveform.squeeze(0).cpu()
        
        # Fallback: gerar waveform temporária (substituir pela lógica real)
        # Em produção, usar o codec correto do Qwen3-TTS
        return torch.randn(24000)  # 1 segundo de ruído temporário