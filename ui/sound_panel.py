# ui/sound_panel.py
# ─── Painel de eventos sonoros para o modo Cinema ─────────────────────────────

import customtkinter as ctk
from tkinter import messagebox
from pathlib import Path


def display_sound_events(app, sound_events_per_segment: list):
    """
    Renderiza no painel de sons todos os eventos detetados.
    sound_events_per_segment: lista paralela a app.segments com dicts {sounds, music}.
    """
    # Limpar painel
    for w in app.sound_scroll.winfo_children():
        w.destroy()

    total_sounds = sum(len(e.get("sounds", [])) for e in sound_events_per_segment)
    total_music  = sum(1 for e in sound_events_per_segment if e.get("music"))

    ctk.CTkLabel(
        app.sound_scroll,
        text=f"🔊 {total_sounds} efeitos sonoros   🎵 {total_music} momentos musicais detetados",
        font=("Roboto", 12, "bold"), text_color="#e67e22"
    ).pack(pady=(5, 8))

    for i, (seg, ev) in enumerate(zip(app.segments, sound_events_per_segment)):
        sounds = ev.get("sounds", [])
        music  = ev.get("music")
        if not sounds and not music:
            continue

        seg_frame = ctk.CTkFrame(app.sound_scroll, fg_color="#1a1a2e")
        seg_frame.pack(fill="x", padx=5, pady=2)

        # Cabeçalho do segmento
        preview = seg.get("text", "")[:60].replace("\n", " ")
        ctk.CTkLabel(
            seg_frame,
            text=f"[{i+1}] {preview}…",
            font=("Consolas", 10), text_color="#7f8c8d", anchor="w"
        ).pack(fill="x", padx=8, pady=(4, 2))

        # Efeitos sonoros
        for ev_s in sounds:
            _sound_row(seg_frame, ev_s, i, app)

        # Música
        if music:
            _music_row(seg_frame, music, i, app)


def _sound_row(parent, ev: dict, seg_idx: int, app):
    row = ctk.CTkFrame(parent, fg_color="#16213e")
    row.pack(fill="x", padx=10, pady=1)
    row.grid_columnconfigure(2, weight=1)

    # Ícone + nome
    ctk.CTkLabel(row, text="🔊", width=24).grid(row=0, column=0, padx=(6, 2))

    name_var = ctk.StringVar(value=ev["sound"])
    name_entry = ctk.CTkEntry(row, textvariable=name_var, width=160,
                               font=("Consolas", 11))
    name_entry.grid(row=0, column=1, padx=4, pady=3)
    # Guardar referência para leitura posterior
    ev["_name_var"] = name_var

    # Descrição PT
    ctk.CTkLabel(row, text=ev.get("description_pt", ""),
                 font=("Roboto", 10), text_color="#95a5a6",
                 anchor="w").grid(row=0, column=2, padx=6, sticky="ew")

    # Duração
    ctk.CTkLabel(row, text=f"⏱ {ev['duration']:.1f}s",
                 font=("Roboto", 10), text_color="#3498db",
                 width=60).grid(row=0, column=3, padx=4)

    # Posição
    pos_var = ctk.StringVar(value=ev.get("position", "during"))
    ctk.CTkOptionMenu(row, values=["before", "during", "after"],
                      variable=pos_var, width=90).grid(row=0, column=4, padx=4)
    ev["_pos_var"] = pos_var

    # Volume
    vol_var = ctk.IntVar(value=ev.get("volume", -10))
    vol_slider = ctk.CTkSlider(row, from_=-30, to=0, variable=vol_var, width=80)
    vol_slider.grid(row=0, column=5, padx=4)
    ev["_vol_var"] = vol_var

    # Indicador DB
    from cinema.sound_db import get_db
    found = get_db().find(ev["sound"]) is not None
    status_text = "✅" if found else "❌ não encontrado"
    status_color = "#2ecc71" if found else "#e74c3c"
    ctk.CTkLabel(row, text=status_text, text_color=status_color,
                 font=("Roboto", 10), width=100).grid(row=0, column=6, padx=4)


def _music_row(parent, music: dict, seg_idx: int, app):
    row = ctk.CTkFrame(parent, fg_color="#0d1b2a")
    row.pack(fill="x", padx=10, pady=1)
    row.grid_columnconfigure(2, weight=1)

    ctk.CTkLabel(row, text="🎵", width=24).grid(row=0, column=0, padx=(6, 2))

    name_var = ctk.StringVar(value=music.get("music", ""))
    ctk.CTkEntry(row, textvariable=name_var, width=160,
                 font=("Consolas", 11)).grid(row=0, column=1, padx=4, pady=3)
    music["_name_var"] = name_var

    ctk.CTkLabel(row, text="Música de fundo",
                 font=("Roboto", 10), text_color="#9b59b6",
                 anchor="w").grid(row=0, column=2, padx=6, sticky="ew")

    ctk.CTkLabel(row, text=f"⏱ {music.get('music_duration', 10.0):.0f}s",
                 font=("Roboto", 10), text_color="#3498db",
                 width=60).grid(row=0, column=3, padx=4)

    vol_var = ctk.IntVar(value=music.get("music_volume", -20))
    ctk.CTkSlider(row, from_=-40, to=-5, variable=vol_var,
                  width=80).grid(row=0, column=5, padx=4)
    music["_vol_var"] = vol_var

    from cinema.sound_db import get_db
    found = get_db().find(music.get("music", "")) is not None
    status_text = "✅" if found else "❌ não encontrado"
    status_color = "#2ecc71" if found else "#e74c3c"
    ctk.CTkLabel(row, text=status_text, text_color=status_color,
                 font=("Roboto", 10), width=100).grid(row=0, column=6, padx=4)


def build_sound_panel(app):
    """Cria o painel de sons (inicialmente oculto, mostrado apenas no modo Cinema)."""
    app.sound_frame_outer = ctk.CTkFrame(app)

    ctk.CTkLabel(
        app.sound_frame_outer,
        text="🎬 Eventos Sonoros Detetados (Modo Cinema)",
        font=("Roboto", 14, "bold"), text_color="#e67e22"
    ).pack(pady=(8, 3))

    # Barra de ferramentas do painel
    toolbar = ctk.CTkFrame(app.sound_frame_outer)
    toolbar.pack(fill="x", padx=10, pady=3)

    app.btn_analyze_sounds = ctk.CTkButton(
        toolbar, text="🔊 Re-analisar Sons",
        command=app.start_sound_analysis,
        fg_color="#d35400", width=160, height=32
    )
    app.btn_analyze_sounds.pack(side="left", padx=5)

    ctk.CTkButton(
        toolbar, text="📁 Gerir Pasta de Sons",
        command=app._open_sounds_folder,
        fg_color="#555", width=160, height=32
    ).pack(side="left", padx=5)

    app.label_sounds_db = ctk.CTkLabel(
        toolbar, text="", font=("Roboto", 10), text_color="#888"
    )
    app.label_sounds_db.pack(side="left", padx=10)

    app.sound_scroll = ctk.CTkScrollableFrame(app.sound_frame_outer, height=220)
    app.sound_scroll.pack(padx=10, fill="x")

    # Placeholder inicial
    app.sound_placeholder = ctk.CTkLabel(
        app.sound_scroll,
        text="Clica em 'Re-analisar Sons' após a análise do livro.",
        text_color="#555", font=("Roboto", 11)
    )
    app.sound_placeholder.pack(pady=30)
