# ui/app.py
# ─── Janela principal da aplicação ────────────────────────────────────────────

import os
import asyncio
import threading
import logging
from pathlib import Path
from datetime import datetime

import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

from config.settings import TEMP_DIR, OLLAMA_BASE_URL, DEFAULT_NARRATOR
from core.extractor import extract_text
from core.analysis_cache import (
    get_analysis_path, compute_book_hash,
    save_analysis, load_analysis
)
from core.ollama_analyzer import (
    get_ollama_models, warmup_ollama,
    split_into_blocks, sanitize_segments, analyze_block
)
from tts.engine import TTSEngine
from tts.exporter import create_m4b

logger = logging.getLogger(__name__)


class AudiobookApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Portugal Narrator Qwen3 Pro v7.3 - Dual Model")
        self.geometry("1100x1100")

        self.ollama_base_url = OLLAMA_BASE_URL
        self.ollama_url = f"{self.ollama_base_url}/api/generate"
        self.model_name = "qwen2.5:7b"
        self.file_path = ""
        self.cover_path = ""
        self.temp_dir = TEMP_DIR

        self.characters = {}
        self.segments = []
        self.raw_text = ""
        self.current_analysis_file = None

        self.tts = TTSEngine(temp_dir=self.temp_dir, log_fn=self.log)
        self.voice_cache = {}

        self._build_ui()

    # ═══════════════════════════════════════════════════════════════════════════
    # UI
    # ═══════════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        from ui.widgets import (
            build_header, build_file_section, build_ollama_section,
            build_action_section, build_character_section,
            build_audio_controls, build_progress_section
        )
        self.grid_columnconfigure(0, weight=1)
        build_header(self)
        build_file_section(self)
        build_ollama_section(self)
        build_action_section(self)
        build_character_section(self)
        build_audio_controls(self)
        build_progress_section(self)

    # ─── Helpers UI ──────────────────────────────────────────────────────────
    def log(self, text: str):
        self.after(0, self._log_safe, text)

    def _log_safe(self, text: str):
        self.textbox.insert("end", f"  > {text}\n")
        self.textbox.see("end")

    def set_progress(self, value: float, label: str = ""):
        self.after(0, self._progress_safe, value, label)

    def _progress_safe(self, value: float, label: str):
        self.progress_bar.set(value)
        self.label_progress.configure(text=label)

    def update_speed_label(self, val):
        self.label_speed_val.configure(text=f"{float(val):.2f}x")

    def select_file(self):
        file = filedialog.askopenfilename(filetypes=[("Livros", "*.txt;*.pdf;*.epub")])
        if file:
            self.file_path = file
            self.label_file_info.configure(text=f"📖 {os.path.basename(file)}", text_color="#3498db")
            self.entry_title.delete(0, tk.END)
            self.entry_title.insert(0, Path(file).stem)

            analysis_path = get_analysis_path(file)
            if analysis_path.exists():
                self.log(f"💡 Análise anterior detetada: {analysis_path.name}")
                if messagebox.askyesno("Análise Encontrada",
                                       f"Já existe uma análise para este livro.\n\n"
                                       f"Ficheiro: {analysis_path.name}\n"
                                       f"Data: {datetime.fromtimestamp(analysis_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M')}\n\n"
                                       f"Queres carregar esta análise?"):
                    self._load_analysis_data(str(analysis_path))

    def select_cover(self):
        file = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg;*.jpeg;*.png")])
        if file:
            self.cover_path = file

    def refresh_models(self):
        self.available_models = get_ollama_models(self.ollama_base_url)
        self.model_combobox.configure(values=self.available_models)
        self.log(f"✅ Modelos Ollama atualizados: {len(self.available_models)}")

    # ─── Carregar análise ────────────────────────────────────────────────────
    def _load_analysis_data(self, analysis_path: str):
        data = load_analysis(analysis_path)
        if not data:
            messagebox.showerror("Erro", "Não foi possível carregar a análise.")
            return

        if self.file_path and "book_hash" in data:
            current_text = extract_text(self.file_path)
            if compute_book_hash(current_text) != data["book_hash"]:
                self.log("⚠️ AVISO: O conteúdo do livro mudou desde a análise!")
                if not messagebox.askyesno("Conteúdo Alterado",
                                           "O conteúdo do livro mudou.\n\nContinuar com a análise antiga?"):
                    return

        self.characters = data.get("characters", {})
        self.segments = sanitize_segments(data.get("segments", []))
        self.current_analysis_file = analysis_path

        if self.file_path and not self.raw_text:
            self.raw_text = extract_text(self.file_path)

        self.log(f"📂 Análise carregada: {Path(analysis_path).name}")
        self.log(f"   📊 {len(self.characters)} personagens, {len(self.segments)} segmentos")
        self.log(f"   📅 Criada em: {data.get('created_at', 'desconhecido')}")

        self.after(0, self._display_characters)
        self.after(0, lambda: self.btn_generate.configure(state="normal"))

    def load_analysis_from_file(self):
        file = filedialog.askopenfilename(
            title="Selecionar Análise",
            filetypes=[("Análises JSON", "*.analysis.json"), ("JSON", "*.json"), ("Todos", "*.*")],
            initialdir=str(Path("analyses"))
        )
        if file:
            try:
                import json
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                book_file = data.get("book_file", "")
                if book_file and Path(book_file).exists() and not self.file_path:
                    self.file_path = book_file
                    self.label_file_info.configure(
                        text=f"📖 {os.path.basename(book_file)}", text_color="#3498db")
                    self.entry_title.delete(0, tk.END)
                    self.entry_title.insert(0, Path(book_file).stem)
            except Exception:
                pass
            self._load_analysis_data(file)

    # ═══════════════════════════════════════════════════════════════════════════
    # FASE 1: ANÁLISE
    # ═══════════════════════════════════════════════════════════════════════════
    def start_analysis(self):
        if not self.file_path:
            messagebox.showwarning("Atenção", "Seleciona um livro primeiro.")
            return
        self.btn_analyze.configure(state="disabled")
        self.btn_generate.configure(state="disabled")
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _run_analysis(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._analyze_book())
        except Exception as e:
            logger.error(f"Erro na análise: {e}")
            self.log(f"❌ Erro: {e}")
        finally:
            self.after(0, lambda: self.btn_analyze.configure(state="normal"))

    async def _analyze_book(self):
        self.log("🔥 Aquecendo Ollama...")
        self.model_name = self.model_combobox.get()
        warmup_ollama(self.ollama_url, self.model_name)

        self.log("📖 A extrair texto do livro...")
        self.set_progress(0.05, "A extrair texto...")
        self.raw_text = extract_text(self.file_path)
        if not self.raw_text:
            self.log("❌ Texto vazio.")
            return

        blocks = split_into_blocks(self.raw_text, max_chars=4000)
        self.log(f"📊 Texto dividido em {len(blocks)} blocos para análise.")

        all_characters, all_segments = {}, []
        for i, block in enumerate(blocks):
            self.set_progress(0.1 + 0.7 * (i / len(blocks)),
                              f"Analisando bloco {i+1}/{len(blocks)}...")
            self.log(f"🔍 Analisando bloco {i+1}/{len(blocks)} ({len(block)} chars)...")

            context = "\n".join(blocks[max(0, i-1):i])
            result = await analyze_block(
                self.ollama_url, self.model_name, block, context, all_characters
            )
            if result:
                for cid, cdata in result.get("characters", {}).items():
                    if cid not in all_characters:
                        all_characters[cid] = cdata
                all_segments.extend(result.get("segments", []))

        self.characters = all_characters
        self.segments = sanitize_segments(all_segments)
        self.log(f"✨ Análise concluída: {len(self.characters)} personagens, {len(self.segments)} segmentos.")

        if "narrator" not in self.characters:
            self.characters["narrator"] = DEFAULT_NARRATOR.copy()

        saved = save_analysis(
            self.file_path, self.raw_text,
            self.characters, self.segments, self.model_name
        )
        if saved:
            self.current_analysis_file = saved
            self.log(f"💾 Análise guardada: {Path(saved).name}")

        self.after(0, self._display_characters)
        self.after(0, lambda: self.btn_generate.configure(state="normal"))
        self.set_progress(1.0, "Análise concluída! Configura as vozes e clica em GERAR.")

    # ═══════════════════════════════════════════════════════════════════════════
    # FASE 2: EXIBIR PERSONAGENS
    # ═══════════════════════════════════════════════════════════════════════════
    def _display_characters(self):
        from ui.character_panel import display_characters
        display_characters(self)

    # ═══════════════════════════════════════════════════════════════════════════
    # FASE 3: GERAR AUDIOBOOK
    # ═══════════════════════════════════════════════════════════════════════════
    def start_generation(self):
        if not self.segments:
            messagebox.showwarning("Atenção", "Analisa o livro primeiro.")
            return
        self.btn_generate.configure(state="disabled")
        self.btn_analyze.configure(state="disabled")
        threading.Thread(target=self._run_generation, daemon=True).start()

    def _run_generation(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._generate_audiobook())
        except Exception as e:
            logger.error(f"Erro na geração: {e}")
            self.log(f"❌ Erro: {e}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            self.after(0, lambda: self.btn_generate.configure(state="normal"))
            self.after(0, lambda: self.btn_analyze.configure(state="normal"))

    async def _generate_audiobook(self):
        self.log("🧬 A preparar vozes e a garantir consistência...")

        await self.tts.load_base()
        await self.tts.load_voicedesign()

        for cid, cdata in self.characters.items():
            # Capturar descrição da UI antes de gerar âncora
            if cid in self.char_widgets:
                widget_data = self.characters[cid]
                if "_desc_entry" in widget_data:
                    cdata["description"] = widget_data["_desc_entry"].get()
            await self.tts.ensure_anchor(cid, cdata)

        self.log(f"🎙️ Vozes fixadas. A gerar {len(self.segments)} segmentos...")
        audio_sequence = []
        s_para = self.tts.create_silence(0.6, "s_para.wav")

        for i, seg in enumerate(self.segments):
            self.set_progress(i / len(self.segments), f"Segmento {i+1}/{len(self.segments)}")

            if not isinstance(seg, dict):
                continue
            text    = seg.get("text", "").strip()
            cid     = seg.get("character_id", "narrator")
            emotion = seg.get("emotion", "neutral")
            pace    = float(seg.get("pace", 1.0)) * float(self.speed_slider.get())

            if not text:
                continue

            cdata = self.characters.get(cid, self.characters.get("narrator", {}))
            out_wav = self.temp_dir / f"seg_{i:05d}.wav"
            self.log(f"  [{i+1}/{len(self.segments)}] {cdata.get('name')} ({emotion}): {text[:40]}...")

            success = await asyncio.to_thread(
                self.tts.clone_with_emotion,
                text, cdata.get("ref_audio"), emotion, pace, str(out_wav),
                cdata.get("ref_text", "")   # transcrição da âncora (ICL) ou "" (x_vector)
            )
            if success:
                audio_sequence.append(out_wav)
                audio_sequence.append(s_para)

        if audio_sequence:
            title  = self.entry_title.get() or "Audiobook"
            author = self.entry_author.get() or "IA"
            ok = create_m4b(audio_sequence, title, author, self.cover_path, log_fn=self.log)
            if ok:
                self.after(0, lambda: messagebox.showinfo(
                    "Sucesso", f"Audiobook criado:\n{title}.m4b"))
