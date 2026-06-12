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
            self.segments   = sanitize_segments(all_segments)

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
        self.log("🔍 A mapear segmentos em cache (verificação ultra-rápida)...")
        cached_indices = set()
        chars_with_uncached_segments = set()

        for i, seg in enumerate(self.segments):
            if isinstance(seg, dict):
                # Usa a verificação ultra-rápida que acabámos de corrigir
                if self.tts.is_segment_cached(i, seg.get("text", "")):
                    cached_indices.add(i)
                else:
                    # Este segmento precisa de ser gerado. Quem é o personagem?
                    # O teu JSON usa a chave "character_id"
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

        chars_to_prepare = {}
        if mode == "narrator":
            chars_to_prepare = {"narrator": self.characters.get("narrator", DEFAULT_NARRATOR.copy())}
        else:
            for cid in chars_with_uncached_segments:
                if cid in self.characters:
                    chars_to_prepare[cid] = self.characters[cid]
                    # Atualizar descrição da UI se o utilizador a tiver editado
                    if "_desc_entry" in self.characters.get(cid, {}):
                        chars_to_prepare[cid]["description"] = self.characters[cid]["_desc_entry"].get()

        # ── FASE 3: GERAR ÂNCORAS (Agora sim, apenas para os filtrados) ─────
        need_design_for_anchors = any(not c.get("ref_audio") for c in chars_to_prepare.values())
        if need_design_for_anchors:
            await self.tts.load_voicedesign()

        for cid, cdata in chars_to_prepare.items():
            await self.tts.ensure_anchor(cid, cdata)

        # ── Fase 4: Decidir quais modelos carregar para síntese ─────────────
        needs_base = self.tts.needs_base(chars_to_prepare)
        needs_vd   = self.tts.needs_voicedesign(chars_to_prepare)
        n_fallback = sum(1 for c in chars_to_prepare.values() if not c.get("ref_audio"))
        
        if n_fallback:
            needs_vd = True
            self.log(f"   ⚠️ {n_fallback} personagem(ns) sem âncora → usarão VoiceDesign direto.")

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

            if not success:
                self.log(f"   ❌ Segmento {i+1} falhou → ignorado.")
                failed_segments.append(i)
                continue

            final_wav = str(out_wav)
            if mode == "cinema":
                sound_data = (self.sound_events[i] if i < len(self.sound_events) else {"sounds": [], "music": None})
                sound_data = _read_sound_panel_values(sound_data)
                if sound_data.get("sounds") or sound_data.get("music"):
                    from cinema.mixer import apply_cinema_mix
                    final_wav = await asyncio.to_thread(apply_cinema_mix, final_wav, sound_data, self.temp_dir, i, self.log)

            audio_sequence.append(final_wav)
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