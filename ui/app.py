# ui/app.py
# ── Janela principal — suporta modos Narrador / Novela / Cinema ──────────────
# Integração Lógica Alexandria: Pós-processamento universal, deteção avançada
# de discurso, resolução de coreferência por turnos, pausas dinâmicas por capítulo
# e cache com validação de texto.
import os
import re
import json
import asyncio
import threading
import logging
import subprocess
import sys
import requests
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

from config.settings import (
    TEMP_DIR, OLLAMA_BASE_URL, DEFAULT_NARRATOR,
    PRODUCTION_MODES, PRODUCTION_MODE_IDS,
)
from core.extractor import extract_text
from core.analysis_cache import (
    get_analysis_path, compute_book_hash,
    save_analysis, load_analysis
)
from core.ollama_analyzer import (
    get_ollama_models, warmup_ollama,
    split_into_blocks, sanitize_segments,
    analyze_block, extract_aliases,
    NameMapper, resolve_generic_ids,
    map_emotion
)
from core.post_processor import post_process_analysis_universal
from tts.engine import TTSEngine
from tts.exporter import create_m4b
from cinema.sound_analyzer import analyze_sounds_batch
from cinema.sound_db import get_db
from cinema.mixer import apply_cinema_mix
from core.ollama_utils import wait_for_model_async

logger = logging.getLogger(__name__)


# Descrições dos modos para a UI
MODE_DESCRIPTIONS = {
    "narrator": (
        "Uma única voz narra todo o texto. "
        "Ideal para leituras simples, poesia ou documentários."
    ),
    "novela": (
        "Cada personagem tem a sua própria voz. "
        "O Ollama identifica automaticamente quem fala em cada segmento."
    ),
    "cinema": (
        "Novela completa com vozes múltiplas + efeitos sonoros + música. "
        "O Ollama deteta os sons descritos no texto e adiciona-os no momento exato."
    ),
}


class AudiobookApp(ctk.CTk):
    # ui/app.py – dentro da classe AudiobookApp

    def __init__(self):
        super().__init__()
        self.title("Portugal Narrator Qwen3 Pro v7.5 - Alexandria Logic")
        self.geometry("1100x1200")

        self.ollama_base_url = OLLAMA_BASE_URL
        self.ollama_url = f"{self.ollama_base_url}/api/generate"
        self.model_name = "gemma3:27b"
         # ── GARANTIR QUE O SERVIDOR OLLAMA ESTÁ ARRANCADO ──────────────
        from core.ollama_utils import ensure_ollama_server
        if not ensure_ollama_server(self.ollama_base_url):
            self.after(1000, lambda: messagebox.showerror(
                "Erro Ollama", 
                "Não foi possível conectar ou arrancar o servidor Ollama.\n\n"
                "Certifica-te que o Ollama está instalado e que o comando 'ollama' funciona no terminal."
            ))
        self.file_path = ""
        self.cover_path = ""
        self.temp_dir = TEMP_DIR

        self.characters = {}
        self.segments = []
        self.raw_text = ""
        self.current_analysis_file = None
        self.sound_events = []
        self.aliases = {}
        self.user_settings = {}

        self.tts = TTSEngine(temp_dir=self.temp_dir, log_fn=lambda msg: self.log(msg))
        self.voice_cache = {}
        self._stop_event = threading.Event()
        self._production_mode = "novela"

        # ── CONSTRUIR UI ──────────────────────────────────────────────────────────
        self._build_ui()
        self._create_menu()

        # ── CARREGAR DEFINIÇÕES DO UTILIZADOR ────────────────────────────────────
        from ui.settings_window import load_settings
        self.user_settings = load_settings()

        # ── APLICAR DEFINIÇÕES (inclui modelo) ──────────────────────────────────
        self._apply_settings(self.user_settings)

        # ── ATUALIZAR MODELOS DISPONÍVEIS ───────────────────────────────────────
        self.refresh_models()

        # ── LOG DE INICIALIZAÇÃO ─────────────────────────────────────────────────
        self.log(f"🚀 Aplicação iniciada. Modelo: {self.model_name}")
        self.log(f"⚙️ Definições: block_size={self.user_settings.get('max_block_size', 6000)}")

    def _create_menu(self):
        menu_bar = tk.Menu(self)
        self.config(menu=menu_bar)
        settings_menu = tk.Menu(menu_bar, tearoff=0)
        settings_menu.add_command(label="Definições", command=self._open_settings)
        menu_bar.add_cascade(label="⚙️", menu=settings_menu)

    def _apply_settings(self, new_settings):
        """Aplica as definições do utilizador aos módulos relevantes."""
        import core.ollama_analyzer as oa

        # ── APLICAR TIMEOUTS E TAMANHO DO BLOCO ──────────────────────────────
        oa.MAX_BLOCK_SIZE = new_settings.get("max_block_size", oa.MAX_BLOCK_SIZE)
        oa.TIMEOUT_SMALL = new_settings.get("ollama_timeout_small", 120)
        oa.TIMEOUT_MEDIUM = new_settings.get("ollama_timeout_medium", 300)
        oa.TIMEOUT_LARGE = new_settings.get("ollama_timeout_large", 600)

        self.user_settings = new_settings

        # ── ATUALIZAR MODELO ──────────────────────────────────────────────────────
        saved_model = new_settings.get("ollama_model", "gemma3:27b")

        # Garantir que a lista de modelos está atualizada
        self.available_models = get_ollama_models(self.ollama_base_url)
        if not self.available_models:
            self.available_models = ["gemma3:27b"]

        # Verificar se o modelo guardado está disponível
        if saved_model in self.available_models:
            self.model_name = saved_model
        else:
            self.log(f"⚠️ Modelo '{saved_model}' não disponível. A usar '{self.available_models[0]}'.")
            self.model_name = self.available_models[0]

        # Atualizar a combobox da UI (se já existir)
        if hasattr(self, 'model_combobox'):
            self.model_combobox.configure(values=self.available_models)
            self.model_combobox.set(self.model_name)

        self.log(f"✅ Definições aplicadas: modelo={self.model_name}, "
                f"block_size={new_settings.get('max_block_size', 6000)}, "
                f"batch_size={new_settings.get('ollama_batch_size', 3)}")

    def _open_settings(self):
        from ui.settings_window import SettingsWindow, load_settings
        current = load_settings()
        SettingsWindow(self, current, self._apply_settings)

    # ═══════════════════════════════════════════════════════════════════════════
    # UI - CONSTRUÇÃO
    # ═══════════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        from ui.widgets import (
            build_header, build_file_section, build_production_mode_section,
            build_ollama_section, build_action_section, build_character_section,
            build_audio_controls, build_progress_section
        )
        from ui.sound_panel import build_sound_panel

        self.grid_columnconfigure(0, weight=1)
        build_header(self)
        build_file_section(self)
        build_production_mode_section(self)
        build_ollama_section(self)
        build_action_section(self)
        build_character_section(self)
        build_sound_panel(self)
        build_audio_controls(self)
        build_progress_section(self)

        self._update_mode_visibility()

    def _on_production_mode_changed(self, selected_label: str):
        idx = PRODUCTION_MODES.index(selected_label)
        self._production_mode = PRODUCTION_MODE_IDS[idx]
        desc = MODE_DESCRIPTIONS.get(self._production_mode, "")
        self.label_mode_desc.configure(text=desc)
        self._update_mode_visibility()

    def _update_mode_visibility(self):
        mode = self._production_mode
        if not hasattr(self, "char_scroll"):
            return

        if mode == "narrator":
            self.char_scroll.pack_forget()
        else:
            self.char_scroll.pack(padx=20, fill="x")

        if not hasattr(self, "sound_frame_outer"):
            return

        if mode == "cinema":
            self.sound_frame_outer.pack(pady=5, padx=20, fill="x")
        else:
            self.sound_frame_outer.pack_forget()

    # ═══════════════════════════════════════════════════════════════════════════
    # UI - HELPERS
    # ═══════════════════════════════════════════════════════════════════════════
    def log(self, text: str):
        self.after(0, self._log_safe, text)

    def _log_safe(self, text: str):
        self.textbox.insert("end", f"    > {text}\n")
        self.textbox.see("end")

    def set_progress(self, value: float, label: str = ""):
        self.after(0, self._progress_safe, value, label)

    def _progress_safe(self, value: float, label: str):
        self.progress_bar.set(value)
        self.label_progress.configure(text=label)

    def update_speed_label(self, val):
        self.label_speed_val.configure(text=f"{float(val):.2f}x")

    # ═══════════════════════════════════════════════════════════════════════════
    # UI - SELEÇÃO DE FICHEIROS
    # ═══════════════════════════════════════════════════════════════════════════
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

    def _open_sounds_folder(self):
        from config.settings import SOUNDS_DIR
        SOUNDS_DIR.mkdir(exist_ok=True)
        if sys.platform == "win32":
            subprocess.Popen(["explorer", str(SOUNDS_DIR)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(SOUNDS_DIR)])
        else:
            subprocess.Popen(["xdg-open", str(SOUNDS_DIR)])
        from cinema.sound_db import get_db
        db = get_db()
        db.reload()
        self.label_sounds_db.configure(text=f"DB: {db.total} sons disponíveis")

    def refresh_models(self):
        self.available_models = get_ollama_models(self.ollama_base_url)
        self.model_combobox.configure(values=self.available_models)
        self.log(f"✅ Modelos Ollama atualizados: {len(self.available_models)}")

    # ═══════════════════════════════════════════════════════════════════════════
    # CARREGAR ANÁLISE
    # ═══════════════════════════════════════════════════════════════════════════
    def _load_analysis_data(self, analysis_path: str):
        data = load_analysis(analysis_path)
        if not data:
            messagebox.showerror("Erro", "Não foi possível carregar a análise.")
            return

        if self.file_path and "book_hash" in data:
            current_text = extract_text(self.file_path)
            if compute_book_hash(current_text) != data["book_hash"]:
                self.log("⚠️ AVISO: O conteúdo do livro mudou desde a análise!")
                if not messagebox.askyesno("Conteúdo Alterado", "O conteúdo do livro mudou. Continuar com a análise antiga?"):
                    return

        self.characters = data.get("characters", {})
        self.segments = sanitize_segments(data.get("segments", []))
        self.aliases = data.get("aliases", {})
        self.current_analysis_file = analysis_path
        self.sound_events = data.get("sound_events", [])

        if self.file_path and not self.raw_text:
            self.raw_text = extract_text(self.file_path)

        self.log(f"📂 Análise carregada: {Path(analysis_path).name}")
        self.log(f"   📊 {len(self.characters)} personagens, {len(self.segments)} segmentos")
        self.log(f"   🔗 {len(self.aliases)} conjuntos de aliases")

        self.after(0, self._display_characters)
        self.after(0, self._display_sound_events)
        self.after(0, lambda: self.btn_generate.configure(state="normal"))

    def load_analysis_from_file(self):
        file = filedialog.askopenfilename(
            title="Selecionar Análise",
            filetypes=[("Análises JSON", "*.analysis.json"), ("JSON", "*.json"), ("Todos", "*.*")],
            initialdir=str(Path("analyses"))
        )
        if file:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    d = json.load(f)
                book_file = d.get("book_file", "")
                if book_file and Path(book_file).exists() and not self.file_path:
                    self.file_path = book_file
                    self.label_file_info.configure(text=f"📖 {os.path.basename(book_file)}", text_color="#3498db")
                    self.entry_title.delete(0, tk.END)
                    self.entry_title.insert(0, Path(book_file).stem)
            except Exception:
                pass
            self._load_analysis_data(file)

    # ═══════════════════════════════════════════════════════════════════════════
    # FASE 1: ANÁLISE DO LIVRO (LÓGICA ALEXANDRIA + PÓS-PROCESSAMENTO UNIVERSAL)
    # ═══════════════════════════════════════════════════════════════════════════
    def _smart_segment_splitter(self, segments: list) -> list:
        """Pós-processamento inteligente para separar discurso direto de narração."""
        refined_segments = []
        for seg in segments:
            if not isinstance(seg, dict):
                refined_segments.append(seg)
                continue

            text = seg.get("text", "").strip()
            character_id = seg.get("character_id", "narrator")
            emotion = seg.get("emotion", "neutral")

            try:
                pace = float(seg.get("pace", 1.0))
            except (ValueError, TypeError):
                pace = 1.0

            try:
                pause_ms = int(seg.get("pause_ms", 0))
            except (ValueError, TypeError):
                pause_ms = 0

            refined_segments.extend(self._split_discourse_and_narration(text, character_id, emotion, pace, pause_ms))

        return self._detect_and_insert_chapter_pauses(refined_segments)

    def _infer_emotion_from_context(self, full_text: str, quote_position: int, quote_text: str) -> str:
        """Inferir emoção a partir de verbos dicendi, pontuação e palavras‑chave."""
        context_window = 200
        start = max(0, quote_position - context_window)
        end = min(len(full_text), quote_position + len(quote_text) + context_window)
        context = full_text[start:end]

        # 1. Verbos dicendi + emoção implícita
        verb_map = {
            r'\bgritou\b': 'angry',
            r'\bberrou\b': 'angry',
            r'\btrovejou\b': 'angry',
            r'\bsussurrou\b': 'whisper',
            r'\bcochichou\b': 'whisper',
            r'\bmurmurou\b': 'whisper',
            r'\bexclamou\b': 'joyful',
            r'\bsoluçou\b': 'sad',
            r'\bdisse com lágrimas\b': 'sad',
            r'\btremia\b': 'fearful',
            r'\bcom medo\b': 'fearful',
            r'\bcalmamente\b': 'calm',
            r'\bserenamente\b': 'calm',
            r'\bnervosamente\b': 'tense',
            r'\bhesitante\b': 'tense',
            r'\bcom raiva\b': 'angry',
            r'\birado\b': 'angry',
        }
        for pattern, emotion in verb_map.items():
            if re.search(pattern, context, re.IGNORECASE):
                return emotion

        # 2. Pontuação no próprio diálogo
        if '!!' in quote_text or '! !' in quote_text:
            return 'joyful' if 'alegria' in context.lower() or 'feliz' in context.lower() else 'angry'
        if '...' in quote_text and len(quote_text) < 50:
            return 'sad'
        if '?' in quote_text:
            return 'tense'

        # 3. Palavras de intensidade
        intensidade = {
            r'\boh\b|\bauau\b|\buau\b': 'joyful',
            r'\bai\b|\bmeu deus\b': 'fearful',
            r'\bcaramba\b|\bdiabo\b': 'angry',
        }
        for pattern, emotion in intensidade.items():
            if re.search(pattern, context, re.IGNORECASE):
                return emotion

        return 'neutral'

    def _split_discourse_and_narration(self, text: str, character_id: str, emotion: str, pace: float, pause_ms: int) -> list:
        """
        Separa discurso direto de narração. Suporta Aspas, Guillemets e Travessões (PT).
        """
        # Se não for narrador e não houver marcadores de fala, devolve o segmento intacto
        if character_id != "narrator" and not re.search(r'["«»"]|—', text):
            return [{"text": text, "character_id": character_id, "emotion": emotion, "pace": pace, "pause_ms": pause_ms}]

        # Apenas o narrador é dividido em partes (narração + diálogo)
        if character_id == "narrator":
            result = []
            quote_pattern = re.compile(r'(["«])(.*?)(["»])', re.DOTALL)
            dash_pattern = re.compile(r'(?:^|\n)\s*—\s*(.*?)\s*(?:—|\n|$)', re.DOTALL)

            all_matches = []
            # Encontrar aspas/guillemets
            for m in quote_pattern.finditer(text):
                all_matches.append({
                    'start': m.start(),
                    'end': m.end(),
                    'text': m.group(2).strip(),
                    'type': 'quote'
                })
            # Encontrar travessões (que não estejam dentro de aspas)
            for m in dash_pattern.finditer(text):
                if not any(m.start() >= om['start'] and m.start() < om['end'] for om in all_matches):
                    all_matches.append({
                        'start': m.start(),
                        'end': m.end(),
                        'text': m.group(1).strip(),
                        'type': 'dash'
                    })

            all_matches.sort(key=lambda x: x['start'])

            # Se não houver diálogo, devolve o texto como narração
            if not all_matches:
                return [{"text": text, "character_id": "narrator", "emotion": emotion, "pace": pace, "pause_ms": pause_ms}]

            last_end = 0
            for match in all_matches:
                # --- Texto antes do diálogo (narração) ---
                before = text[last_end:match['start']].strip()
                if before:
                    # A primeira pausa pode ser mais longa (início do bloco), as restantes mais curtas
                    pause_before = pause_ms if not result else 150
                    result.append({
                        "text": before,
                        "character_id": "narrator",
                        "emotion": emotion,
                        "pace": pace,
                        "pause_ms": pause_before
                    })

                # --- Diálogo ---
                dialog_text = match['text']
                if dialog_text:
                    # 1. Inferir emoção a partir do contexto
                    inferred_emotion = self._infer_emotion_from_context(text, match['start'], dialog_text)

                    # 2. Se o Ollama já atribuiu uma emoção explícita (não‑neutral), prevalece
                    if emotion not in ['neutral', '']:
                        inferred_emotion = emotion

                    # 3. Identificar o falante
                    speaker = self._identify_speaker_from_context(text, dialog_text, match['start'], result)

                    # 4. Adicionar segmento de fala
                    result.append({
                        "text": dialog_text,
                        "character_id": speaker,
                        "emotion": self._adjust_emotion_for_discourse(inferred_emotion),
                        "pace": pace * 1.05,      # ligeiramente mais rápido para fala
                        "pause_ms": 300           # pausa padrão após fala
                    })

                last_end = match['end']

            # --- Texto após o último diálogo (narração) ---
            after = text[last_end:].strip()
            if after:
                result.append({
                    "text": after,
                    "character_id": "narrator",
                    "emotion": emotion,
                    "pace": pace,
                    "pause_ms": 200
                })

            return result if result else [{"text": text, "character_id": "narrator", "emotion": emotion, "pace": pace, "pause_ms": pause_ms}]

        # Caso seja uma personagem com marcadores de fala (fallback seguro)
        return [{"text": text, "character_id": character_id, "emotion": emotion, "pace": pace, "pause_ms": pause_ms}]

    def _identify_speaker_from_context(self, full_text: str, quote_text: str, quote_position: int, current_result: list) -> str:
        """Identifica quem fala usando verbos dicendi, aliases e contexto."""
        search_window = 300
        start = max(0, quote_position - search_window)
        end = min(len(full_text), quote_position + len(quote_text) + search_window)
        context = full_text[start:end]

        # 1. Procurar verbos dicendi + nome
        dicendi_verbs = ["disse", "exclamou", "respondeu", "gritou", "sussurrou",
                         "afirmou", "perguntou", "murmurou", "berrou", "cochichou",
                         "declarou", "retrucou", "anunciou", "continuou", "acrescentou", "comentou"]

        # Construir mapa de personagens
        char_map = {cid: cdata.get("name", "").lower() for cid, cdata in self.characters.items()}
        name_to_id = {v: k for k, v in char_map.items() if v}

        # Procurar padrões como "disse Asa" ou "Asa disse"
        for cid, cdata in self.characters.items():
            if cid == "narrator":
                continue
            name = cdata.get("name", "")
            if not name:
                continue
            if re.search(r'\b' + re.escape(name) + r'\b', context, re.IGNORECASE):
                for verb in dicendi_verbs:
                    pattern = rf'(?:{re.escape(name)}\s+{verb}|{verb}\s+{re.escape(name)})'
                    if re.search(pattern, context, re.IGNORECASE):
                        return cid

        # 2. Usar o NameMapper para termos genéricos
        if self.aliases:
            mapper = NameMapper(self.aliases)
            # Procurar por termos genéricos no contexto
            generic_terms = ["pai", "mãe", "mae", "senhor", "senhora", "rapaz", "rapariga", "ele", "ela", "eles", "elas"]
            for generic in generic_terms:
                if re.search(r'\b' + re.escape(generic) + r'\b', context, re.IGNORECASE):
                    mapped = mapper.map_generic(generic, context)
                    if mapped != "narrator":
                        return mapped

        # 3. Se houver uma personagem no último segmento (alternância de turnos)
        for seg in reversed(current_result):
            if seg.get("character_id") and seg["character_id"] != "narrator":
                return seg["character_id"]

        return "narrator"

    def _adjust_emotion_for_discourse(self, base_emotion: str) -> str:
        """Ajusta a emoção para discurso direto, mantendo a inferida."""
        # Se já temos uma emoção não‑neutral, mantém‑na
        if base_emotion not in ['neutral', '']:
            return base_emotion
        # Caso contrário, usa 'calm' como padrão para fala
        return 'calm'

    def _detect_and_insert_chapter_pauses(self, segments: list) -> list:
        """Deteta cabeçalhos de capítulo e insere pausas longas."""
        chapter_pattern = re.compile(r'(?i)^\s*(cap[ií]tulo\s+[ivx\d]+|parte\s+[ivx\d]+|livro\s+[ivx\d]+)\s*[:\-]?\s*(.*?)$', re.MULTILINE)
        refined_segments = []
        for seg in segments:
            text = seg.get("text", "")
            match = chapter_pattern.search(text)
            if match and len(text.strip()) < 150:
                refined_segments.append({"text": "", "character_id": "narrator", "emotion": "neutral", "pace": 1.0, "pause_ms": 3000, "is_chapter_break": True})
                refined_segments.append(seg)
                refined_segments.append({"text": "", "character_id": "narrator", "emotion": "neutral", "pace": 1.0, "pause_ms": 1500, "is_chapter_break": True})
            else:
                refined_segments.append(seg)
        return refined_segments

    # ═══════════════════════════════════════════════════════════════════════════
    # INICIAR ANÁLISE
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
        """
        Analisa o livro completo usando Ollama.
        """
        from ui.settings_window import load_settings
        import core.ollama_analyzer as oa

        # ── CARREGAR DEFINIÇÕES DO UTILIZADOR ────────────────────────────────
        self.user_settings = load_settings()
        self.log(f"⚙️ Definições carregadas: block_size={self.user_settings.get('max_block_size', 6000)}, "
                f"batch_size={self.user_settings.get('ollama_batch_size', 3)}")

        # ── APLICAR TIMEOUTS, TAMANHO DO BLOCO E BATCH_SIZE ────────────────
        oa.TIMEOUT_SMALL = self.user_settings.get("ollama_timeout_small", 120)
        oa.TIMEOUT_MEDIUM = self.user_settings.get("ollama_timeout_medium", 300)
        oa.TIMEOUT_LARGE = self.user_settings.get("ollama_timeout_large", 600)
        oa.MAX_BLOCK_SIZE = self.user_settings.get("max_block_size", 6000)

        # ── WARMUP INTELIGENTE ──────────────────────────────────────────────
        from core.ollama_utils import wait_for_model_async

        self.log("🔥 A verificar/carregar o modelo Ollama...")
        self.model_name = self.model_combobox.get()
        warmup_timeout = self.user_settings.get("ollama_warmup_timeout", 300)

        model_ready = await wait_for_model_async(
            self.ollama_base_url,
            self.model_name,
            max_wait=warmup_timeout,
            check_interval=3,
            progress_callback=self.set_progress
        )

        if not model_ready:
            self.log("⚠️ Modelo não carregou a tempo – continuando mesmo assim (pode demorar mais)")
        else:
            self.log("✅ Modelo pronto para análise.")

        # ── EXTRAIR TEXTO ──────────────────────────────────────────────────────
        self.log("📖 A extrair texto do livro...")
        self.set_progress(0.05, "A extrair texto...")
        self.raw_text = extract_text(self.file_path)
        if not self.raw_text:
            self.log("❌ Texto vazio.")
            return

        # ══════════════════════════════════════════════════════════════════════
        # ️ MODO DE TESTE (controlado pela definição)
        # ══════════════════════════════════════════════════════════════════════
        test_mode = self.user_settings.get('test_mode', False)
        if test_mode:
            self.log("⚠️ MODO DE TESTE ATIVO: A limitar o livro...")
            chapter_pattern = re.compile(
                r'(?i)^\s*(cap[ií]tulo|chapter|parte|part|livro|book)\s+[ivx\d]+',
                re.MULTILINE
            )
            matches = list(chapter_pattern.finditer(self.raw_text))
            if len(matches) >= 4:
                self.raw_text = self.raw_text[:matches[3].start()]
                self.log("✂️ Texto truncado com sucesso. A analisar apenas os 3 primeiros capítulos.")
            else:
                self.raw_text = self.raw_text[:30000]
                self.log("✂️ Capítulos não detetados. Texto truncado para 30.000 caracteres.")
        else:
            self.log("📖 Modo de análise completa (desativado).")

        mode = self._production_mode

        # ══════════════════════════════════════════════════════════════════════
        # MODO NARRADOR
        # ══════════════════════════════════════════════════════════════════════
        if mode == "narrator":
            self.characters = {"narrator": DEFAULT_NARRATOR.copy()}
            paragraphs = [p.strip() for p in self.raw_text.split('\n\n') if p.strip()]
            self.segments = [
                {"text": p, "character_id": "narrator",
                "emotion": "neutral", "pace": 1.0, "pause_ms": 0}
                for p in paragraphs
            ]
            self.log(f"✨ Modo Narrador: {len(self.segments)} parágrafos.")
            saved = save_analysis(self.file_path, self.raw_text, self.characters, self.segments, self.model_name)
            if saved:
                self.current_analysis_file = saved
                self.log(f"💾 Análise guardada: {Path(saved).name}")
            self.after(0, self._display_characters)
            self.after(0, lambda: self.btn_generate.configure(state="normal"))
            self.set_progress(1.0, "Análise concluída! Configura as vozes e clica em GERAR.")
            return

        # ══════════════════════════════════════════════════════════════════════
        # MODO NOVELA / CINEMA - ANÁLISE COMPLETA
        # ══════════════════════════════════════════════════════════════════════
        # 1. CRIAR BLOCOS (USANDO O TAMANHO DAS DEFINIÇÕES)
        max_block = self.user_settings.get("max_block_size", 6000)
        if max_block > 25000:
            self.log("⚠️ max_block_size > 25000 pode exceder VRAM. A limitar para 25000.")
            max_block = 25000

        # ── BATCH_SIZE ─────────────────────────────────────────────────────────
        # ATENÇÃO: o Ollama processa pedidos sequencialmente (single-threaded).
        # Um batch_size > 1 não acelera — os pedidos ficam em fila e o timeout
        # começa a contar desde o início, garantindo falhas nos últimos do batch.
        # Para blocos grandes (>4000 chars): forçar batch_size=1.
        # Para blocos pequenos: permitir no máximo 2.
        user_batch_size = self.user_settings.get("ollama_batch_size", 1)
        if max_block > 4000:
            batch_size = 1
            if user_batch_size > 1:
                self.log(f"⚠️ batch_size limitado a 1 (blocos grandes + Ollama single-threaded).")
        else:
            batch_size = min(user_batch_size, 2)

        blocks = split_into_blocks(self.raw_text, max_chars=max_block)
        self.log(f"📊 Texto dividido em {len(blocks)} blocos para análise (máx. {max_block} chars, batch_size={batch_size}).")

        # SE MODO DE TESTE ESTIVER ATIVO, LIMITAR O NÚMERO DE BLOCOS PARA TESTE RÁPIDO
        if test_mode and len(blocks) > 50:
            self.log(f"⚠️ MODO DE TESTE: Ainda são {len(blocks)} blocos. A limitar aos primeiros 50 para teste rápido...")
            blocks = blocks[:50]
            self.log(f"✅ Limitado a {len(blocks)} blocos.")

        # 2. LOOP DE ANÁLISE
        all_characters, all_segments = {}, []

        for i in range(0, len(blocks), batch_size):
            batch = blocks[i:i + batch_size]
            batch_tasks = []

            for j, block in enumerate(batch):
                block_index = i + j
                self.set_progress(0.1 + 0.6 * (block_index / len(blocks)),
                                f"Analisando bloco {block_index+1}/{len(blocks)}...")
                self.log(f"🔍 Bloco {block_index+1}/{len(blocks)} ({len(block)} chars)...")
                context = "\n".join(blocks[max(0, block_index-1):block_index])

                task = analyze_block(
                    self.ollama_url,
                    self.model_name,
                    block,
                    context,
                    all_characters,
                    self.aliases,
                    user_settings=self.user_settings
                )
                batch_tasks.append((block_index, task))

            batch_results = await asyncio.gather(*[task for _, task in batch_tasks], return_exceptions=True)

            for (block_index, _), result in zip(batch_tasks, batch_results):
                if isinstance(result, Exception):
                    self.log(f"⚠️ Erro no bloco {block_index+1}: {result}")
                    continue
                if result and isinstance(result, dict):
                    raw_chars = result.get("characters", {})
                    if isinstance(raw_chars, dict):
                        for cid, cdata in raw_chars.items():
                            cid = str(cid).strip().lower().replace(" ", "_")
                            if cid and cid not in all_characters:
                                all_characters[cid] = cdata

                    raw_segs = result.get("segments", [])
                    block_len = len(blocks[block_index])
                    n_segs = len(raw_segs) if isinstance(raw_segs, list) else 0
                    # Alertar se o Ollama devolveu poucos segmentos para o tamanho do bloco
                    # (heurística: esperamos ≥1 segmento por 500 chars)
                    expected_min = max(1, block_len // 500)
                    if n_segs < expected_min:
                        self.log(f"   ⚠️ Bloco {block_index+1}: apenas {n_segs} segmento(s) para {block_len} chars (esperado ≥{expected_min}). Possível timeout/truncamento.")
                    else:
                        self.log(f"   ✅ Bloco {block_index+1}: {n_segs} segmentos extraídos.")

                    if isinstance(raw_segs, list):
                        all_segments.extend(raw_segs)

        # 3. SANITIZAÇÃO
        self.log("🧹 A sanitizar segmentos...")
        all_segments = sanitize_segments(all_segments)

        # 4. PÓS‑PROCESSAMENTO UNIVERSAL
        self.log("🔄 A aplicar pós-processamento universal...")
        from core.post_processor import post_process_analysis_universal

        chars, segs = post_process_analysis_universal(
            all_characters,
            all_segments,
            normalize=True,
            merge_duplicates=True,
            enhance_descriptions=True
        )
        self.characters = chars
        self.segments = segs

        # 5. GARANTIR NARRADOR
        if "narrator" not in self.characters:
            self.characters["narrator"] = DEFAULT_NARRATOR.copy()

        # 6. EXTRAIR ALIASES
        self.log("🔗 A extrair aliases para mapear termos genéricos...")
        self.aliases = await extract_aliases(
            self.ollama_url,
            self.model_name,
            self.raw_text,
            self.characters,
            user_settings=self.user_settings
        )
        self.log(f"   📝 {len(self.aliases)} conjuntos de aliases extraídos")

        # 7. RESOLVER CONFLITOS DE ALIASES
        from core.post_processor import resolve_alias_conflicts
        self.aliases = resolve_alias_conflicts(self.aliases)
        self.log(f"   🧹 Aliases após remoção de conflitos: {len(self.aliases)} conjuntos")

        # 8. RESOLVER IDs GENÉRICOS
        if self.aliases:
            self.log("🔄 A resolver IDs genéricos (pai, mãe, etc.)...")
            self.segments = resolve_generic_ids(self.segments, self.characters, self.aliases)
            self.log(f"   ✅ {len(self.segments)} segmentos processados")

        # 9. SPLITTER INTELIGENTE
        self.log("✂️ A aplicar splitter inteligente...")
        self.segments = self._smart_segment_splitter(self.segments)

        # 10. DEDUPLICAÇÃO DE SEGMENTOS
        # Usar apenas o texto como chave mas permitir repetições que estejam
        # separadas por mais de 3 segmentos (frases que se repetem no livro são válidas).
        # O que queremos evitar é o mesmo segmento ser emitido duas vezes consecutivas
        # pelo Ollama (artefacto de geração), não frases repetidas no texto original.
        unique_segments = []
        recent_texts = []   # janela deslizante dos últimos 5 textos
        for seg in self.segments:
            text = seg.get("text", "").strip()
            if not text:
                continue
            if text not in recent_texts:
                unique_segments.append(seg)
                recent_texts.append(text)
                if len(recent_texts) > 5:
                    recent_texts.pop(0)
            # Se o texto já apareceu recentemente (duplicado consecutivo), ignorar
        removed = len(self.segments) - len(unique_segments)
        self.segments = unique_segments
        self.log(f"   🧹 Segmentos únicos: {len(self.segments)} ({removed} duplicados consecutivos removidos)")

        # 11. BLINDAGEM FINAL
        for seg in self.segments:
            if seg.get("character_id") not in self.characters:
                seg["character_id"] = "narrator"

        # ══════════════════════════════════════════════════════════════════════
        # 🔄 REVISÃO AUTOMÁTICA DE ATRIBUIÇÃO DE FALAS (usando Ollama)
        # ══════════════════════════════════════════════════════════════════════
        if self.user_settings.get('auto_revise', True):
            try:
                from core.analysis_reviser import revise_analysis
                self.log("🔄 A aplicar revisão automática de atribuição de falas e verificação de omissões...")
                self.segments, corrections, missing = revise_analysis(
                    self.segments,
                    self.characters,
                    self.ollama_base_url,
                    self.model_name,
                    original_text=self.raw_text,
                    max_segments=200,
                    log_fn=self.log
                )
                if corrections:
                    self.log(f"   ✅ Aplicadas {len(corrections)} correções de falas.")
                else:
                    self.log("   ✅ Nenhuma correção de falas necessária.")
                if missing:
                    self.log(f"   ⚠️ Foram detetados {len(missing)} possíveis omissões de texto.")
            except Exception as e:
                self.log(f"   ⚠️ Revisão automática falhou: {e}")

        self.log(f"✨ Análise concluída: {len(self.characters)} personagens, {len(self.segments)} segmentos.")

        # 12. GUARDAR ANÁLISE
        saved = save_analysis(self.file_path, self.raw_text, self.characters, self.segments, self.model_name)
        if saved:
            self.current_analysis_file = saved
            self.log(f"💾 Análise guardada: {Path(saved).name}")

            if self.aliases:
                try:
                    with open(saved, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    data['aliases'] = self.aliases
                    with open(saved, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    self.log(f"   🔗 Aliases adicionados ao ficheiro de análise.")
                except Exception as e:
                    self.log(f"   ⚠️ Não foi possível guardar aliases: {e}")

        # 13. ATUALIZAR UI
        self.after(0, self._display_characters)
        self.after(0, self._display_sound_events)
        self.after(0, lambda: self.btn_generate.configure(state="normal"))
        self.set_progress(1.0, "Análise concluída! Configura as vozes e clica em GERAR.")

        if mode == "cinema":
            self.after(500, self.start_sound_analysis)

    # ═══════════════════════════════════════════════════════════════════════════
    # FASE 1b: ANÁLISE DE SONS (apenas Cinema)
    # ═══════════════════════════════════════════════════════════════════════════
    def start_sound_analysis(self):
        if not self.segments:
            messagebox.showwarning("Atenção", "Analisa o livro primeiro.")
            return
        self.btn_analyze_sounds.configure(state="disabled")
        threading.Thread(target=self._run_sound_analysis, daemon=True).start()

    def _run_sound_analysis(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._analyze_sounds())
        except Exception as e:
            logger.error(f"Erro na análise de sons: {e}")
            self.log(f"❌ Erro sons: {e}")
        finally:
            self.after(0, lambda: self.btn_analyze_sounds.configure(state="normal"))

    async def _analyze_sounds(self):
        db = get_db()
        self.log(f"🔊 A analisar sons em {len(self.segments)} segmentos... (DB: {db.total} sons)")
        self.label_sounds_db.configure(text=f"DB: {db.total} sons disponíveis")

        self.sound_events = await analyze_sounds_batch(self.ollama_url, self.model_name, self.segments, progress_fn=self.set_progress)

        total_sfx = sum(len(e.get("sounds", [])) for e in self.sound_events)
        total_music = sum(1 for e in self.sound_events if e.get("music"))
        self.log(f"✅ Sons detetados: {total_sfx} efeitos, {total_music} momentos musicais.")

        if self.current_analysis_file:
            try:
                with open(self.current_analysis_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["sound_events"] = self.sound_events
                with open(self.current_analysis_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                self.log("💾 Eventos sonoros guardados na análise.")
            except Exception as e:
                self.log(f"⚠️ Erro ao guardar sons: {e}")

        self.after(0, self._display_sound_events)
        self.set_progress(1.0, "Análise de sons concluída!")

    # ═══════════════════════════════════════════════════════════════════════════
    # FASE 2: EXIBIR PERSONAGENS / SONS
    # ═══════════════════════════════════════════════════════════════════════════
    def _display_characters(self):
        if self._production_mode == "narrator":
            return
        from ui.character_panel import display_characters
        display_characters(self)

    def _display_sound_events(self):
        if self._production_mode != "cinema" or not self.sound_events:
            return
        from ui.sound_panel import display_sound_events
        display_sound_events(self, self.sound_events)

    # ═══════════════════════════════════════════════════════════════════════════
    # FASE 3: GERAR AUDIOBOOK
    # ═══════════════════════════════════════════════════════════════════════════
    def start_generation(self):
        if not self.segments:
            messagebox.showwarning("Atenção", "Analisa o livro primeiro.")
            return
        self._stop_event.clear()
        self.btn_generate.configure(state="disabled")
        self.btn_analyze.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        threading.Thread(target=self._run_generation, daemon=True).start()

    def _run_generation(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._generate_audiobook())
        except Exception as e:
            logger.error(f"Erro na geração: {e}")
            self.log(f"❌ Erro: {e}")
        finally:
            self.after(0, self._on_generation_finished)

    def _on_generation_finished(self):
        self.btn_generate.configure(state="normal")
        self.btn_analyze.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        if self._stop_event.is_set():
            self.log("⛔ Geração interrompida pelo utilizador.")

    def stop_generation(self):
        self._stop_event.set()
        self.log("⛔ A parar após o segmento actual...")
        self.btn_stop.configure(state="disabled")

    async def _generate_audiobook(self):
        from tts.vram_manager import unload_ollama, log_vram
        from config.settings import OLLAMA_UNLOAD_AFTER_ANALYSIS

        mode = self._production_mode
        self.log(f"🎬 Modo: {mode.upper()}")

        if OLLAMA_UNLOAD_AFTER_ANALYSIS:
            unload_ollama(self.ollama_base_url, self.model_combobox.get(), self.log)
            log_vram(self.log)

        self.log(f"🔍 A mapear segmentos em cache na pasta: {self.tts.temp_dir.absolute()}")
        total = len(self.segments)
        cached_indices = set()
        chars_with_uncached_segments = set()

        for i, seg in enumerate(self.segments):
            if not isinstance(seg, dict):
                continue
            if self.tts.is_segment_cached(i, seg.get("text", "")):
                cached_indices.add(i)
            else:
                cid = seg.get("character_id", "narrator") if mode != "narrator" else "narrator"
                chars_with_uncached_segments.add(cid)

        n_cached = len(cached_indices)
        self.log(f"⚡ Cache: {n_cached}/{total} segmentos já prontos e válidos.")
        if n_cached == total:
            self.log("✅ Todos os segmentos em cache. A saltar para concatenação...")
            await self._concatenate_and_finish(total, n_cached, [], [])
            return

        self.log(f"🧬 A preparar vozes APENAS para {len(chars_with_uncached_segments)} personagem(ns)...")
        max_chars = 200
        char_list = list(chars_with_uncached_segments)
        if len(char_list) > max_chars:
            if "narrator" in char_list:
                char_list = ["narrator"] + [c for c in char_list if c != "narrator"][:max_chars-1]
            else:
                char_list = char_list[:max_chars]

        chars_to_prepare = {}
        if mode == "narrator":
            chars_to_prepare = {"narrator": self.characters.get("narrator", DEFAULT_NARRATOR.copy())}
        else:
            for cid in char_list:
                if cid in self.characters:
                    chars_to_prepare[cid] = self.characters[cid]
                    if "_desc_entry" in self.characters.get(cid, {}):
                        chars_to_prepare[cid]["description"] = self.characters[cid]["_desc_entry"].get()

        self.log("🇵🇹 Forçando sotaque português europeu para todas as personagens...")
        for cid, cdata in chars_to_prepare.items():
            current_desc = cdata.get("description", "")
            if "portugal" not in current_desc.lower() and "português" not in current_desc.lower():
                cdata["description"] = f"{current_desc} Voz portuguesa de Portugal, sotaque europeu."

        chars_needing_anchors = {cid: cdata for cid, cdata in chars_to_prepare.items() if not cdata.get("ref_audio")}
        if chars_needing_anchors:
            self.log(f"⚓ Gerando âncoras para {len(chars_needing_anchors)} personagem(ns)...")
            await self.tts.load_voicedesign()
            for cid, cdata in chars_needing_anchors.items():
                if self._stop_event.is_set():
                    return
                await self.tts.ensure_anchor(cid, cdata)

        needs_base = self.tts.needs_base(chars_to_prepare)
        needs_vd = self.tts.needs_voicedesign(chars_to_prepare)
        if self._stop_event.is_set():
            return

        if needs_base and not needs_vd:
            self.tts.release_voicedesign()
        if needs_base:
            await self.tts.load_base()
        if needs_vd:
            await self.tts.load_voicedesign()
        log_vram(self.log)

        self.log(f"🎙️ A gerar {total - n_cached} segmento(s)...")
        audio_sequence = []
        failed_segments = []

        for i, seg in enumerate(self.segments):
            if self._stop_event.is_set():
                break
            self.set_progress(i / total, f"Segmento {i+1}/{total}")
            if not isinstance(seg, dict):
                continue

            if seg.get("is_chapter_break"):
                seg_pause_s = max(0.3, min(5.0, seg.get("pause_ms", 600) / 1000.0))
                silence_file = self.tts.create_silence(seg_pause_s, f"s_pause_{int(seg_pause_s*10)}.wav")
                audio_sequence.append(str(silence_file))
                continue

            if i in cached_indices:
                audio_sequence.append(str(self.tts.segment_cache_path(i)))
                seg_pause_s = max(0.3, min(5.0, seg.get("pause_ms", 600) / 1000.0))
                silence_file = self.tts.create_silence(seg_pause_s, f"s_pause_{int(seg_pause_s*10)}.wav")
                audio_sequence.append(str(silence_file))
                continue

            text = seg.get("text", "").strip()
            if not text:
                continue

            emotion = seg.get("emotion", "neutral")
            pace = float(seg.get("pace", 1.0)) * float(self.speed_slider.get())
            seg_cid = seg.get("character_id", "narrator") if mode != "narrator" else "narrator"
            cdata = self.characters.get(seg_cid, self.characters.get("narrator", {}))
            out_wav = self.tts.segment_cache_path(i)
            name = cdata.get("name", "?")

            self.log(f"  [{i+1}] {name} ({emotion}): {text[:40]}...")

            use_clone = bool(cdata.get("ref_audio"))
            if use_clone and self.tts.model_base is None:
                await self.tts.load_base()
            elif not use_clone and self.tts.model_design is None:
                await self.tts.load_voicedesign()

            success = await asyncio.to_thread(
                self.tts.clone_with_emotion, text, cdata.get("ref_audio"), emotion, pace * 0.8, str(out_wav), "", cdata.get("description", "Voz neutra, português de Portugal.")
            )

            if success:
                self.tts.save_segment_metadata(i, text, emotion, pace)

                final_wav = str(out_wav)
                if mode == "cinema":
                    sound_data = self.sound_events[i] if i < len(self.sound_events) else {"sounds": [], "music": None}
                    sound_data = self._read_sound_panel_values(sound_data)
                    if sound_data.get("sounds") or sound_data.get("music"):
                        final_wav = await asyncio.to_thread(apply_cinema_mix, final_wav, sound_data, self.temp_dir, i, self.log)

                if self._validate_audio_file(final_wav, i+1):
                    audio_sequence.append(final_wav)
                    seg_pause_s = max(0.3, min(5.0, seg.get("pause_ms", 600) / 1000.0))
                    silence_file = self.tts.create_silence(seg_pause_s, f"s_pause_{int(seg_pause_s*10)}.wav")
                    audio_sequence.append(str(silence_file))
                else:
                    failed_segments.append(i)
            else:
                failed_segments.append(i)
                try:
                    Path(out_wav).unlink(missing_ok=True)
                except:
                    pass

        self.tts.release_base()
        self.tts.release_voicedesign()
        log_vram(self.log)
        await self._concatenate_and_finish(total, n_cached, failed_segments, audio_sequence)

    async def _concatenate_and_finish(self, total: int, n_cached: int, failed: list, audio_seq: list):
        generated = total - n_cached - len(failed)
        self.log(f"📊 Gerados: {generated} | Cache: {n_cached} | Falhas: {len(failed)} | Total: {total}")
        if audio_seq:
            title = self.entry_title.get() or "Audiobook"
            author = self.entry_author.get() or "IA"
            ok = await asyncio.to_thread(create_m4b, audio_seq, title, author, self.cover_path, self.log)
            if ok:
                self.after(0, lambda: messagebox.showinfo("Sucesso", f"Audiobook criado:\n{title}.m4b"))

    def _validate_audio_file(self, file_path: str, segment_index: int) -> bool:
        try:
            import soundfile as sf
            path = Path(file_path)
            if not path.exists() or path.stat().st_size < 512:
                return False
            audio, sr = sf.read(str(path))
            duration = len(audio) / sr if sr > 0 else 0
            return duration >= 0.05
        except Exception as e:
            self.log(f"   ⚠️ Erro ao validar {segment_index}: {e}")
            return False

    # ─── Utilitário: lê valores editados pelo utilizador no painel de sons ───────
    def _read_sound_panel_values(self, sound_data: dict) -> dict:
        result = {"sounds": [], "music": sound_data.get("music")}
        for ev in sound_data.get("sounds", []):
            new_ev = dict(ev)
            if "_name_var" in ev:
                new_ev["sound"] = ev["_name_var"].get()
            if "_pos_var" in ev:
                new_ev["position"] = ev["_pos_var"].get()
            if "_vol_var" in ev:
                new_ev["volume"] = ev["_vol_var"].get()
            result["sounds"].append(new_ev)
        music = sound_data.get("music")
        if music:
            new_music = dict(music)
            if "_name_var" in music:
                new_music["music"] = music["_name_var"].get()
            if "_vol_var" in music:
                new_music["music_volume"] = music["_vol_var"].get()
            result["music"] = new_music
        return result