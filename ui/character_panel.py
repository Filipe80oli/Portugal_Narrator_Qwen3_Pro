# ui/character_panel.py
# ─── Painel de personagens: exibição, edição e preview de voz ─────────────────

import asyncio
import subprocess
from tkinter import filedialog, messagebox
import customtkinter as ctk


def display_characters(app):
    """Renderiza os widgets de personagem no painel lateral."""
    if app.char_placeholder.winfo_exists():
        app.char_placeholder.destroy()

    for w in app.char_scroll.winfo_children():
        w.destroy()
    app.char_widgets.clear()

    for cid, cdata in app.characters.items():
        frame = ctk.CTkFrame(app.char_scroll)
        frame.pack(fill="x", padx=5, pady=3)
        frame.grid_columnconfigure(3, weight=1)

        name_entry = ctk.CTkEntry(frame, width=130, font=("Roboto", 12, "bold"))
        name_entry.insert(0, cdata.get("name", cid))
        name_entry.grid(row=0, column=0, padx=5, pady=5)

        type_label = ctk.CTkLabel(frame, text=f"[{cdata.get('type', '?')}]",
                                  width=100, text_color="#f39c12", font=("Roboto", 11))
        type_label.grid(row=0, column=1, padx=5, pady=5)

        desc_entry = ctk.CTkEntry(frame, width=380, font=("Roboto", 11))
        desc_entry.insert(0, cdata.get("description", "Voz neutra"))
        desc_entry.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        voice_mode = ctk.StringVar(value="design")
        mode_menu = ctk.CTkOptionMenu(
            frame, values=["design", "clone"],
            variable=voice_mode, width=120,
            command=lambda v, c=cid: _on_voice_mode_changed(app, c, v)
        )
        mode_menu.grid(row=0, column=3, padx=5, pady=5)

        wav_btn = ctk.CTkButton(
            frame, text="🎙️ Selecionar .wav", width=140,
            command=lambda c=cid: _select_wav_for_char(app, c),
            state="disabled"
        )
        wav_btn.grid(row=0, column=4, padx=5, pady=5)

        wav_label = ctk.CTkLabel(frame, text="", width=150, text_color="#888",
                                  font=("Roboto", 10), anchor="w")
        wav_label.grid(row=0, column=5, padx=5, pady=5)

        preview_btn = ctk.CTkButton(
            frame, text="▶", width=40,
            command=lambda c=cid: asyncio.run(_preview_character(app, c))
        )
        preview_btn.grid(row=0, column=6, padx=5, pady=5)

        app.characters[cid]["_name_entry"] = name_entry
        app.characters[cid]["_desc_entry"] = desc_entry
        app.characters[cid]["_voice_mode"] = voice_mode
        app.characters[cid]["_wav_btn"] = wav_btn
        app.characters[cid]["_wav_label"] = wav_label
        if "ref_audio" not in app.characters[cid]:
            app.characters[cid]["ref_audio"] = None
        app.char_widgets[cid] = frame


def _on_voice_mode_changed(app, char_id: str, mode: str):
    cdata = app.characters[char_id]
    if mode == "clone":
        cdata["_wav_btn"].configure(state="normal")
    else:
        cdata["_wav_btn"].configure(state="disabled")
        cdata["ref_audio"] = None
        cdata["_wav_label"].configure(text="")


def _select_wav_for_char(app, char_id: str):
    f = filedialog.askopenfilename(filetypes=[("WAV", "*.wav")])
    if f:
        out_path = app.temp_dir / f"ref_{char_id}_24k.wav"
        try:
            cmd = ['ffmpeg', '-y', '-i', f, '-ar', '24000', '-ac', '1',
                   '-c:a', 'pcm_s16le', str(out_path)]
            subprocess.run(cmd, capture_output=True, check=True, timeout=30)
            app.characters[char_id]["ref_audio"] = str(out_path)
            app.characters[char_id]["_wav_label"].configure(
                text=f"✅ {out_path.name}", text_color="#2ecc71")
            if char_id in app.voice_cache:
                del app.voice_cache[char_id]
        except Exception as e:
            app.log(f"❌ Erro ao preparar áudio: {e}")


async def _preview_character(app, char_id: str):
    if not app.tts.model_base and not app.tts.model_design:
        app.log("⚠️ Carrega o modelo Qwen3-TTS primeiro (clica em GERAR uma vez).")
        return

    cdata = app.characters[char_id]
    name     = cdata["_name_entry"].get()
    desc     = cdata["_desc_entry"].get()
    mode     = cdata["_voice_mode"].get()
    ref_audio = cdata.get("ref_audio")

    if mode == "clone" and not ref_audio:
        messagebox.showwarning("Atenção", "Seleciona um ficheiro .wav para clonagem.")
        return

    preview_text = f"Olá, eu sou o {name}. Esta é uma amostra da minha voz."
    out_path = app.temp_dir / f"preview_{char_id}.wav"
    app.log(f"▶ A gerar preview de {name}...")

    def _gen():
        try:
            if mode == "clone":
                app.tts.generate_clone(preview_text, ref_audio, str(out_path),
                                       ref_text=cdata.get("ref_text", ""))
            else:
                app.tts.generate_design(preview_text, desc, "neutral", str(out_path))
            subprocess.Popen(['ffplay', '-nodisp', '-autoexit',
                              '-loglevel', 'quiet', str(out_path)])
            app.log(f"✅ Preview de {name} gerado.")
        except Exception as e:
            app.log(f"❌ Erro no preview: {e}")

    await asyncio.to_thread(_gen)
