#!/usr/bin/env python3
# ─── REQUISITOS: pip install customtkinter PyMuPDF ebooklib beautifulsoup4 requests torch soundfile qwen-tts ───
# ─── SISTEMA: FFmpeg no PATH. Ollama em execução. RTX 3090 recomendada. ───
# ─── VERSÃO: 7.3 - Dual Model (Base clone + VoiceDesign) ───

import warnings
import logging
import customtkinter as ctk

from ui.app import AudiobookApp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
warnings.filterwarnings('ignore', category=UserWarning)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

if __name__ == "__main__":
    app = AudiobookApp()
    app.mainloop()
