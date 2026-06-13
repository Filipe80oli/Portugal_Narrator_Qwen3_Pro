# ─── Janela principal — suporta modos Narrador / Novela / Cinema ──────────────
import os
import asyncio
import threading
import logging
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
    split_into_blocks, sanitize_segments, analyze_block
)
from tts.engine import TTSEngine
from tts.exporter import create_m4b

logger = logging.getLogger(__name__)

# Descrições dos modos para a UI
MODE_DESCRIPTIONS = {
    "narrator": (
        "Uma única voz narra todo o texto. "
        "Ideal para leituras simples, poesia ou documentários."
    ),
    "novela":   (
        "Cada personagem tem a sua própria voz. "
        "O Ollama identifica automaticamente quem fala em cada segmento."
    ),
    "cinema":   (
        "Novela completa com vozes múltiplas + efeitos sonoros + música. "
        "O Ollama deteta os sons descritos no texto e adiciona-os no momento exato."
    ),
}

class AudiobookApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Portugal Narrator Qwen3 Pro v7.4 - Dual Model")
        self.geometry("1100x1200")

        self.ollama_base_url = OLLAMA_BASE_URL
        self.ollama_url      = f"{self.ollama_base_url}/api/generate"
        self.model_name      = "qwen2.5:7b"
        self.file_path       = ""
        self.cover_path      = ""
        self.temp_dir        = TEMP_DIR

        self.characters  = {}
        self.segments    = []
        self.raw_text    = ""
        self.current_analysis_file = None

        # Eventos sonoros detetados (lista paralela a self.segments)
        self.sound_events: list[dict] = []

        self.tts         = TTSEngine(temp_dir=self.temp_dir, log_fn=self.log)
        self.voice_cache = {}

        # Modo de produção ativo
        self._production_mode = "novela"

        self._build_ui()

    # ═══════════════════════════════════════════════════════════════════════════
    # UI
    # ═══════════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        from ui.widgets import (
            build_header, build_file_section, build_production_mode_section,
            build_ollama_section, build_action_section,  build_character_section,
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
        build_sound_panel(self)   # cria sound_frame_outer
        build_audio_controls(self)
        build_progress_section(self)

        # Agora todos os widgets existem -- aplicar visibilidade correta
        self._update_mode_visibility()

    def _on_production_mode_changed(self, selected_label: str):
        idx = PRODUCTION_MODES.index(selected_label)
        self._production_mode = PRODUCTION_MODE_IDS[idx]
        desc = MODE_DESCRIPTIONS.get(self._production_mode, "")
        self.label_mode_desc.configure(text=desc)
        self._update_mode_visibility()

    def _update_mode_visibility(self):
        """
        Mostra/oculta paineis conforme o modo de producao.
        Seguro para chamadas antes de todos os widgets estarem criados.
        """
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

    # ─── Helpers UI ──────────────────────────────────────────────────────────
    def log(self, text: str):
        self.after(0, self._log_safe, text)

    def _log_safe(self, text: str):
        self.textbox.insert("end", f"   > {text}\n")
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
            self.label_file_info.configure(
                text=f"📖 {os.path.basename(file)}", text_color="#3498db")
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
        import subprocess, sys
        if sys.platform == "win32":
            subprocess.Popen(["explorer", str(SOUNDS_DIR)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(SOUNDS_DIR)])
        else:
            subprocess.Popen(["xdg-open", str(SOUNDS_DIR)])
        from cinema.sound_db import get_db
        db = get_db()
        db.reload()
        self.label_sounds_db.configure(
            text=f"DB: {db.total} sons disponíveis")

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
        self.segments   = sanitize_segments(data.get("segments", []))
        self.current_analysis_file = analysis_path

        self.sound_events = data.get("sound_events", [])

        if self.file_path and not self.raw_text:
            self.raw_text = extract_text(self.file_path)

        self.log(f"📂 Análise carregada: {Path(analysis_path).name}")
        self.log(f"   📊 {len(self.characters)} personagens, {len(self.segments)} segmentos")
        self.log(f"   📅 Criada em: {data.get('created_at', 'desconhecido')}")
        if self.sound_events:
            self.log(f"   🔊 {sum(len(e.get('sounds',[])) for e in self.sound_events)} eventos sonoros carregados")

        self.after(0, self._display_characters)
        self.after(0, self._display_sound_events)
        self.after(0, lambda: self.btn_generate.configure(state="normal"))

    def load_analysis_from_file(self):
        file = filedialog.askopenfilename(
            title="Selecionar Análise",
            filetypes=[("Análises JSON", "*.analysis.json"),
                       ("JSON", "*.json"), ("Todos", "*.*")],
            initialdir=str(Path("analyses"))
        )
        if file:
            try:
                import json
                with open(file, "r", encoding="utf-8") as f:
                    d = json.load(f)
                book_file = d.get("book_file", "")
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
    # FASE 1: ANÁLISE DO LIVRO
    # ═══════════════════════════════════════════════════════════════════════════
    def _smart_segment_splitter(self, segments: list) -> list:
        """
        Pós-processamento inteligente para separar discurso direto de narração.
        """
        refined_segments = []
        
        for seg in segments:
            if not isinstance(seg, dict):
                refined_segments.append(seg)
                continue
                
            text = seg.get("text", "").strip()
            character_id = seg.get("character_id", "narrator")
            emotion = seg.get("emotion", "neutral")
            pace = seg.get("pace", 1.0)
            pause_ms = seg.get("pause_ms", 0)
            
            # Verificar padrões de discurso direto + narração no mesmo segmento
            refined_segments.extend(self._split_discourse_and_narration(
                text, character_id, emotion, pace, pause_ms
            ))
        
        return refined_segments

    def _split_discourse_and_narration(self, text: str, character_id: str, 
                                    emotion: str, pace: float, pause_ms: int) -> list:
        """
        Separa discurso direto de narração dentro do mesmo segmento.
        """
        import re
        
        # Padrões comuns de discurso direto
        discourse_patterns = [
            r'(["«])(.*?)(["»])',  # "..." ou «...»
            r'(["«])([^"»]+)(["»])',  # Aspas simples ou duplas
            r'(\w+)\s*(exclamou|disse|respondeu|gritou|sussurrou|afirmou|declarou)',  # Verbos de fala
        ]
        
        # Se for o narrador, verificar se há discurso direto embutido
        if character_id == "narrator":
            # Procurar por aspas no texto
            quote_matches = list(re.finditer(r'(["«])(.*?)(["»])', text))
            
            if quote_matches:
                # Separar discurso direto da narração
                result = []
                last_end = 0
                
                for match in quote_matches:
                    # Parte antes da citação (narração)
                    before = text[last_end:match.start()].strip()
                    if before:
                        result.append({
                            "text": before,
                            "character_id": "narrator",
                            "emotion": emotion,
                            "pace": pace,
                            "pause_ms": pause_ms if len(result) == 0 else 0
                        })
                    
                    # A citação (discurso direto)
                    quote_text = match.group(2).strip()
                    if quote_text:
                        # Tentar identificar quem fala (olhar contexto)
                        speaker = self._identify_speaker(text, quote_text)
                        result.append({
                            "text": quote_text,
                            "character_id": speaker,
                            "emotion": self._adjust_emotion_for_discourse(emotion),
                            "pace": pace * 1.1,  # Ligeiramente mais rápido para diálogo
                            "pause_ms": 300 if len(result) > 0 else pause_ms
                        })
                    
                    last_end = match.end()
                
                # Parte depois da última citação (narração)
                after = text[last_end:].strip()
                if after:
                    result.append({
                        "text": after,
                        "character_id": "narrator",
                        "emotion": emotion,
                        "pace": pace,
                        "pause_ms": 200
                    })
                
                return result
        
        # Caso contrário, retornar o segmento original
        return [{
            "text": text,
            "character_id": character_id,
            "emotion": emotion,
            "pace": pace,
            "pause_ms": pause_ms
        }]

    def _identify_speaker(self, full_text: str, quote_text: str) -> str:
        """
        Tenta identificar quem está falando com base no contexto.
        """
        import re
        
        # Procurar por verbos de fala próximos à citação
        speaker_patterns = [
            r'(\w+)\s+(exclamou|disse|respondeu|gritou|sussurrou|afirmou|declarou)',
            r'(exclamou|disse|respondeu|gritou|sussurrou|afirmou|declarou)\s+(\w+)',
        ]
        
        for pattern in speaker_patterns:
            matches = list(re.finditer(pattern, full_text, re.IGNORECASE))
            for match in matches:
                # Verificar se está próximo à citação
                speaker_name = match.group(1) if match.group(1).lower() not in [
                    'exclamou', 'disse', 'respondeu', 'gritou', 'sussurrou', 'afirmou', 'declarou'
                ] else match.group(2)
                
                # Procurar personagem com nome similar
                for cid, cdata in self.characters.items():
                    if speaker_name.lower() in cdata.get("name", "").lower() or \
                    cdata.get("name", "").lower() in speaker_name.lower():
                        return cid
        
        # Fallback: usar personagem mais recente ou narrador
        return "narrator"

    def _adjust_emotion_for_discourse(self, base_emotion: str) -> str:
        """
        Ajusta a emoção para discurso direto.
        """
        discourse_emotions = {
            "neutral": "calm",
            "calm": "calm",
            "tense": "tense",
            "joyful": "joyful",
            "sad": "sad",
            "angry": "angry",
            "fearful": "fearful",
            "whisper": "whisper"
        }
        return discourse_emotions.get(base_emotion, base_emotion)

    
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

        mode = self._production_mode

        if mode == "narrator":
            self.characters = {"narrator": DEFAULT_NARRATOR.copy()}
            paragraphs = [p.strip() for p in self.raw_text.split('\n\n') if p.strip()]
            self.segments = [
                {"text": p, "character_id": "narrator",
                 "emotion": "neutral", "pace": 1.0, "pause_ms": 0}
                for p in paragraphs
            ]
            self.log(f"✨ Modo Narrador: {len(self.segments)} parágrafos.")
        else:
            blocks = split_into_blocks(self.raw_text, max_chars=4000)
            self.log(f"📊 Texto dividido em {len(blocks)} blocos para análise.")

            all_characters, all_segments = {}, []
            for i, block in enumerate(blocks):
                self.set_progress(0.1 + 0.7 * (i / len(blocks)),
                                   f"Analisando bloco {i+1}/{len(blocks)}...")
                self.log(f"🔍 Bloco {i+1}/{len(blocks)} ({len(block)} chars)...")
                context = "\n".join(blocks[max(0, i-1):i])
                result  = await analyze_block(
                    self.ollama_url, self.model_name, block, context, all_characters)
                if result:
                    for cid, cdata in result.get("characters", {}).items():
                        if cid not in all_characters:
                            all_characters[cid] = cdata
                    all_segments.extend(result.get("segments", []))

            self.characters = all_characters
            raw_segments = sanitize_segments(all_segments)
            self.segments = self._smart_segment_splitter(raw_segments)

            if "narrator" not in self.characters:
                self.characters["narrator"] = DEFAULT_NARRATOR.copy()

            self.log(f"✨ Análise concluída: {len(self.characters)} personagens, {len(self.segments)} segmentos.")

        saved = save_analysis(
            self.file_path, self.raw_text,
            self.characters, self.segments, self.model_name
        )
        if saved:
            self.current_analysis_file  = saved
            self.log(f"💾 Análise guardada: {Path(saved).name}")

        self.after(0, self._display_characters)
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
        from cinema.sound_analyzer import analyze_sounds_batch
        from cinema.sound_db import get_db

        db = get_db()
        self.log(f"🔊 A analisar sons em {len(self.segments)} segmentos... (DB: {db.total} sons)")
        self.label_sounds_db.configure(text=f"DB: {db.total} sons disponíveis")

        self.sound_events = await analyze_sounds_batch(
            self.ollama_url, self.model_name,
            self.segments,
            progress_fn=self.set_progress
        )

        total_sfx    = sum(len(e.get("sounds", [])) for e in self.sound_events)
        total_music = sum(1 for e in self.sound_events if e.get("music"))
        self.log(f"✅ Sons detetados: {total_sfx} efeitos, {total_music} momentos musicais.")

        if self.current_analysis_file:
            import json
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
        mode = self._production_mode
        if mode == "narrator":
            return
        from ui.character_panel import display_characters
        display_characters(self)

    def _display_sound_events(self):
        if self._production_mode != "cinema":
            return
        if not self.sound_events:
            return
        from ui.sound_panel import display_sound_events
        display_sound_events(self, self.sound_events)

    # ═══════════════════════════════════════════════════════════════════════════
    # FASE 3: GERAR AUDIOBOOK (FLUXO OTIMIZADO COM CACHE INTELIGENTE)
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
        from tts.vram_manager import unload_ollama, log_vram
        from config.settings import OLLAMA_UNLOAD_AFTER_ANALYSIS

        mode = self._production_mode
        self.log(f"🎬 Modo: {mode.upper()}")

        # ── Fase 0: Libertar VRAM do Ollama ─────────────────────────────────
        if OLLAMA_UNLOAD_AFTER_ANALYSIS:
            unload_ollama(self.ollama_base_url, self.model_combobox.get(), self.log)
            log_vram(self.log)

        # ── FASE 1: MAPEAMENTO INTELIGENTE DE CACHE ─────────────────────────
        self.log(f"🔍 A mapear segmentos em cache na pasta: {self.tts.temp_dir.absolute()}")
        cached_indices = set()
        chars_with_uncached_segments = set()

        for i, seg in enumerate(self.segments):
            if isinstance(seg, dict):
                # Agora esta chamada é instantânea e não rejeita ficheiros válidos
                if self.tts.is_segment_cached(i, seg.get("text", "")):
                    cached_indices.add(i)
                else:
                    # Este segmento precisa de ser gerado. Quem é o personagem?
                    cid = seg.get("character_id", "narrator") if mode != "narrator" else "narrator"
                    chars_with_uncached_segments.add(cid)

        n_cached = len(cached_indices)
        total = len(self.segments)
        self.log(f"⚡ Cache: {n_cached}/{total} segmentos já prontos.")
        if n_cached == total:
            self.log("✅ Todos os segmentos em cache. A saltar para concatenação...")
            await self._concatenate_and_finish(total, n_cached, [], [])
            return

        # ── FASE 2: PREPARAR APENAS AS VOZES NECESSÁRIAS ────────────────────
        self.log(f"🧬 A preparar vozes APENAS para {len(chars_with_uncached_segments)} personagem(ns) que têm segmentos por gerar...")

        # LIMITAR O NÚMERO DE PERSONAGENS PARA EVITAR TIMEOUT (mas manter o narrador)
        max_chars = 15  # Reduzido para melhor performance
        char_list = list(chars_with_uncached_segments)

        if len(char_list) > max_chars:
            # Sempre incluir o narrador
            if "narrator" in char_list:
                char_list = ["narrator"] + [c for c in char_list if c != "narrator"][:max_chars-1]
            else:
                char_list = char_list[:max_chars]
            self.log(f"⚠️ Limitando personagens de {len(chars_with_uncached_segments)} para {len(char_list)} para evitar timeout")

        chars_to_prepare = {}
        if mode == "narrator":
            chars_to_prepare = {"narrator": self.characters.get("narrator", DEFAULT_NARRATOR.copy())}
        else:
            for cid in char_list:
                if cid in self.characters:
                    chars_to_prepare[cid] = self.characters[cid]
                    # Atualizar descrição da UI se o utilizador a tiver editado
                    if "_desc_entry" in self.characters.get(cid, {}):
                        chars_to_prepare[cid]["description"] = self.characters[cid]["_desc_entry"].get()

        # ── FORÇAR SOTAQUE PT-PT PARA TODAS AS PERSONAGENS ────────────────────
        self.log("🇵🇹 Forçando sotaque português europeu para todas as personagens...")
        for cid, cdata in chars_to_prepare.items():
            # Atualizar descrição para garantir sotaque PT-PT
            current_desc = cdata.get("description", "")
            if "portugal" not in current_desc.lower() and "português" not in current_desc.lower():
                if cid == "narrator":
                    cdata["description"] = f"{current_desc} Voz portuguesa de Portugal, sotaque europeu."
                else:
                    cdata["description"] = f"{current_desc} Voz portuguesa de Portugal, sotaque europeu."

        # ── FASE 3: GERAR ÂNCORAS (Agora sim, apenas para os filtrados) ─────
        # Verificar quais personagens precisam de âncoras (não têm ref_audio)
        chars_needing_anchors = {cid: cdata for cid, cdata in chars_to_prepare.items() 
                                if not cdata.get("ref_audio")}

        if chars_needing_anchors:
            self.log(f"⚓ Gerando âncoras para {len(chars_needing_anchors)} personagem(ns)...")
            await self.tts.load_voicedesign()
            
            anchor_success_count = 0
            for cid, cdata in chars_needing_anchors.items():
                await self.tts.ensure_anchor(cid, cdata)
                if cdata.get("ref_audio"):
                    anchor_success_count += 1
            
            self.log(f"   ✅ {anchor_success_count} âncoras geradas com sucesso")
            
            # Atualizar chars_to_prepare com as âncoras geradas
            for cid, cdata in chars_to_prepare.items():
                if cid in chars_needing_anchors and cdata.get("ref_audio"):
                    # Âncora foi gerada com sucesso
                    pass
                elif cid in chars_needing_anchors and not cdata.get("ref_audio"):
                    # Falha na âncora - usar VoiceDesign direto
                    cdata["ref_audio"] = None
                    cdata["ref_text"] = None
                    

        # ── Fase 4: Decidir quais modelos carregar para síntese ─────────────
        needs_base = self.tts.needs_base(chars_to_prepare)
        needs_vd   = self.tts.needs_voicedesign(chars_to_prepare)
        n_fallback = sum(1 for c in chars_to_prepare.values() if not c.get("ref_audio"))

        # Corrigir contagem de fallback (apenas VoiceDesign direto)
        actual_fallback = sum(1 for c in chars_to_prepare.values() if not c.get("ref_audio") or c.get("ref_audio") is None)
        if actual_fallback > 0:
            self.log(f"   ⚠️ {actual_fallback} personagem(ns) sem âncora → usarão VoiceDesign direto.")

        self.log(f"📦 Modelos: {'Base(clone) ' if needs_base else ''}{'VoiceDesign' if needs_vd else ''}")
        if needs_base and not needs_vd:
            self.tts.release_voicedesign()
        if needs_base: await self.tts.load_base()
        if needs_vd:   await self.tts.load_voicedesign()
        log_vram(self.log)

        # ── Fase 5: Loop de geração (salta os que estão em cache) ───────────
        self.log(f"🎙️ A gerar {total - n_cached} segmento(s)...")
        audio_sequence = []
        failed_segments = []
        s_para = self.tts.create_silence(0.6, "s_para.wav")

        for i, seg in enumerate(self.segments):
            self.set_progress(i / total, f"Segmento {i+1}/{total}")
            if not isinstance(seg, dict): continue

            # SALTO INTELIGENTE: Se está em cache, adiciona à sequência e continua
            if i in cached_indices:
                self.log(f"  ⏭️ [{i+1}] Em cache → saltado")
                audio_sequence.append(str(self.tts.segment_cache_path(i)))
                audio_sequence.append(str(s_para))
                continue

            text = seg.get("text", "").strip()
            if not text: continue

            # ── VALIDAÇÃO CRÍTICA: Evitar uso de texto de âncora como conteúdo ───────────
            from config.settings import ANCHOR_TEXT
            # Verificar se o texto é exatamente igual ao texto da âncora
            if text.strip() == ANCHOR_TEXT.strip():
                self.log(f"   ⚠️ Texto do segmento é idêntico ao texto da âncora - corrigindo...")
                # Isso indica um problema na análise - usar um texto mais apropriado
                text = f"{name} está falando."  # Texto genérico de fallback

            # Verificar se o texto parece ser de âncora (padrões comuns)
            anchor_indicators = ["sotaque de lisboa", "portugal", "europeia", "narrador", "minha voz"]
            is_suspicious_anchor = (
                any(indicator in text.lower() for indicator in anchor_indicators) and 
                len(text) < 200 and  # Textos de âncora são geralmente curtos
                "estou a falar" in text.lower()
            )

            if is_suspicious_anchor:
                self.log(f"   ⚠️ Texto suspeito de ser âncora detectado no segmento {i+1}")
                # Tentar recuperar o texto original ou usar um fallback
                original_text = seg.get("text", "").strip()
                if original_text != text and len(original_text) > len(text):
                    # Usar o texto original se for mais longo
                    text = original_text
                    self.log(f"   🔄 Recuperando texto original: {text[:50]}...")
                else:
                    # Fallback - criar um texto genérico
                    text = f"{name} diz: [conteúdo do diálogo]" 
                    self.log(f"   🔄 Usando texto de fallback para evitar âncora")

            # Garantir que o texto tenha conteúdo significativo
            if len(text.strip()) < 5:
                text = f"{name}: {text}" if text else f"{name} está falando."

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
                self.tts.clone_with_emotion,
                text, cdata.get("ref_audio"), emotion, pace, str(out_wav),
                cdata.get("ref_text", ""),
                cdata.get("description", "Voz neutra, português de Portugal.")
            )

            # VALIDAÇÃO ADICIONAL DE QUALIDADE
            if success:
                # Validar qualidade do áudio gerado
                try:
                    from tts.audio_validator import validate_audio
                    q = validate_audio(str(out_wav), text)
                    if not q.ok:
                        self.log(f"   ⚠️ Áudio de baixa qualidade detectado [{i+1}]: {q.reason}")
                        # Tentar regerar com parâmetros diferentes se for problema de ruído
                        if "ruído" in q.reason.lower() or "silêncio" in q.reason.lower() or "rms" in q.reason.lower():
                            self.log(f"   🔄 Tentando regerar segmento {i+1} com parâmetros conservadores...")
                            # Ajustar parâmetros para tentativa mais conservadora
                            success = await asyncio.to_thread(
                                self.tts.clone_with_emotion,
                                text, cdata.get("ref_audio"), emotion, pace * 0.8, str(out_wav),
                                cdata.get("ref_text", ""),
                                cdata.get("description", "Voz neutra, português de Portugal.")
                            )
                            # Validar novamente
                            if success:
                                q2 = validate_audio(str(out_wav), text)
                                if not q2.ok:
                                    self.log(f"   ❌ Segunda tentativa também falhou [{i+1}]: {q2.reason}")
                                    success = False
                        else:
                            success = False
                except Exception as e:
                    self.log(f"   ⚠️ Erro na validação do segmento {i+1}: {e}")
                    success = False

            if not success:
                self.log(f"   ❌ Segmento {i+1} falhou → ignorado.")
                failed_segments.append(i)
                # Remover arquivo problemático
                try:
                    Path(out_wav).unlink(missing_ok=True)
                except:
                    pass
                continue

            final_wav = str(out_wav)
            if mode == "cinema":
                sound_data = (self.sound_events[i] if i < len(self.sound_events) else {"sounds": [], "music": None})
                sound_data = _read_sound_panel_values(sound_data)
                if sound_data.get("sounds") or sound_data.get("music"):
                    from cinema.mixer import apply_cinema_mix
                    final_wav = await asyncio.to_thread(apply_cinema_mix, final_wav, sound_data, self.temp_dir, i, self.log)

            if self._validate_audio_file(final_wav, i+1):
                audio_sequence.append(final_wav)
            else:
                self.log(f"   ⚠️ Segmento {i+1} inválido - ignorado")
                failed_segments.append(i)

            audio_sequence.append(str(s_para))

        # ── Fase 6: Libertar VRAM TTS e Finalizar ───────────────────────────
        self.tts.release_base()
        self.tts.release_voicedesign()
        log_vram(self.log)
        await self._concatenate_and_finish(total, n_cached, failed_segments, audio_sequence)


    async def _concatenate_and_finish(self, total: int, n_cached: int, failed: list, audio_seq: list):
        generated = total - n_cached - len(failed)
        self.log(f"📊 Gerados: {generated} | Cache: {n_cached} | Falhas: {len(failed)} | Total: {total}")
        if failed:
            self.log(f"   ⚠️ Ignorados: {failed[:10]}{'...' if len(failed)>10 else ''}")

        if audio_seq:
            title  = self.entry_title.get() or "Audiobook"
            author = self.entry_author.get() or "IA"
            ok = await asyncio.to_thread(create_m4b, audio_seq, title, author, self.cover_path, self.log)
            if ok:
                self.after(0, lambda: messagebox.showinfo("Sucesso", f"Audiobook criado:\n{title}.m4b"))

    def _validate_audio_file(self, file_path: str, segment_index: int) -> bool:
        """Valida um arquivo de áudio individual antes da concatenação."""
        try:
            from tts.audio_validator import validate_audio
            import soundfile as sf
            from pathlib import Path
            
            path = Path(file_path)
            if not path.exists():
                return False
                
            # Verificação básica de tamanho
            if path.stat().st_size < 512:  # Menos de 512 bytes
                return False
                
            # Tentar ler o arquivo
            audio, sr = sf.read(str(path))
            duration = len(audio) / sr if sr > 0 else 0
            
            # Verificar duração mínima (menos de 50ms é suspeito)
            if duration < 0.05:
                return False
                
            return True
        except Exception as e:
            self.log(f"   ⚠️ Erro ao validar {segment_index}: {e}")
            return False

# ─── Utilitário: lê valores editados pelo utilizador no painel de sons ────────
def _read_sound_panel_values(sound_data: dict) -> dict:
    """Substitui os valores dos eventos sonoros pelos das widgets (se editados)."""
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