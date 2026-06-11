# ui/widgets.py
# ─── Funções de construção dos blocos da UI ───────────────────────────────────

import customtkinter as ctk
from core.ollama_analyzer import get_ollama_models
from config.settings import PRODUCTION_MODES, PRODUCTION_MODE_IDS


def build_header(app):
    ctk.CTkLabel(app, text="Portugal Narrator Qwen3 Pro v7.3",
                 font=("Roboto", 26, "bold")).pack(pady=(15, 3))
    ctk.CTkLabel(app, text="Qwen3-TTS Base (clone) + VoiceDesign (auto) • Cache de Análise",
                 font=("Roboto", 13), text_color="#3498db").pack(pady=(0, 10))


def build_file_section(app):
    top_frame = ctk.CTkFrame(app)
    top_frame.pack(pady=5, padx=20, fill="x")

    meta_frame = ctk.CTkFrame(top_frame)
    meta_frame.pack(side="left", padx=5, fill="x", expand=True)
    app.entry_title = ctk.CTkEntry(meta_frame, placeholder_text="Título do Livro", width=250)
    app.entry_title.grid(row=0, column=0, padx=5, pady=5)
    app.entry_author = ctk.CTkEntry(meta_frame, placeholder_text="Autor", width=200)
    app.entry_author.grid(row=0, column=1, padx=5, pady=5)

    file_frame = ctk.CTkFrame(top_frame)
    file_frame.pack(side="left", padx=5)
    app.btn_file = ctk.CTkButton(file_frame, text="📖 Selecionar Livro",
                                  command=app.select_file, width=160)
    app.btn_file.grid(row=0, column=0, padx=5, pady=5)
    app.btn_cover = ctk.CTkButton(file_frame, text="🖼️ Capa",
                                   command=app.select_cover, width=80)
    app.btn_cover.grid(row=0, column=1, padx=5, pady=5)
    app.label_file_info = ctk.CTkLabel(file_frame, text="Nenhum",
                                        text_color="#888", font=("Roboto", 11))
    app.label_file_info.grid(row=1, column=0, columnspan=2, padx=5)


def build_production_mode_section(app):
    """Selector dos 3 modos de produção com descrição dinâmica."""
    outer = ctk.CTkFrame(app)
    outer.pack(pady=8, padx=20, fill="x")

    ctk.CTkLabel(outer, text="Modo de Produção:",
                 font=("Roboto", 13, "bold")).pack(side="left", padx=12)

    # Segmented button — escolha exclusiva
    app.production_mode_var = ctk.StringVar(value=PRODUCTION_MODES[1])  # default: Novela

    seg_btn = ctk.CTkSegmentedButton(
        outer,
        values=PRODUCTION_MODES,
        variable=app.production_mode_var,
        command=app._on_production_mode_changed,
        font=("Roboto", 13, "bold"),
        height=38,
        selected_color="#8e44ad",
        selected_hover_color="#7d3c98",
    )
    seg_btn.pack(side="left", padx=10)

    # Descrição do modo selecionado
    app.label_mode_desc = ctk.CTkLabel(
        outer, text="", font=("Roboto", 11),
        text_color="#bdc3c7", wraplength=380, justify="left"
    )
    app.label_mode_desc.pack(side="left", padx=14, fill="x", expand=True)

    # Inicializar descrição
    app._on_production_mode_changed(PRODUCTION_MODES[1])


def build_ollama_section(app):
    ollama_frame = ctk.CTkFrame(app)
    ollama_frame.pack(pady=5, padx=20, fill="x")
    ctk.CTkLabel(ollama_frame, text="Modelo Ollama:").grid(row=0, column=0, padx=10, pady=8)

    app.available_models = get_ollama_models(app.ollama_base_url)
    app.model_combobox = ctk.CTkComboBox(ollama_frame, values=app.available_models, width=280)
    default_model = next(
        (m for m in app.available_models if "qwen2.5:7b" in m or "qwen2.5:14b" in m),
        app.available_models[0] if app.available_models else "qwen2.5:7b"
    )
    app.model_combobox.set(default_model)
    app.model_name = default_model
    app.model_combobox.grid(row=0, column=1, padx=10, pady=8)
    ctk.CTkButton(ollama_frame, text="🔄", width=40,
                  command=app.refresh_models).grid(row=0, column=2, padx=5)


def build_action_section(app):
    action_frame = ctk.CTkFrame(app)
    action_frame.pack(pady=10, padx=20, fill="x")

    app.btn_analyze = ctk.CTkButton(
        action_frame, text="1️⃣ ANALISAR LIVRO",
        command=app.start_analysis, fg_color="#2980b9", height=50,
        font=("Roboto", 14, "bold")
    )
    app.btn_analyze.pack(side="left", padx=5, expand=True, fill="x")

    app.btn_load_analysis = ctk.CTkButton(
        action_frame, text="📂 CARREGAR ANÁLISE",
        command=app.load_analysis_from_file, fg_color="#8e44ad", height=50,
        font=("Roboto", 14, "bold")
    )
    app.btn_load_analysis.pack(side="left", padx=5, expand=True, fill="x")

    app.btn_generate = ctk.CTkButton(
        action_frame, text="2️⃣ GERAR AUDIOBOOK",
        command=app.start_generation, fg_color="#27ae60", height=50,
        font=("Roboto", 14, "bold"), state="disabled"
    )
    app.btn_generate.pack(side="left", padx=5, expand=True, fill="x")


def build_character_section(app):
    ctk.CTkLabel(app, text="🎭 Personagens Detetados & Configuração de Voz",
                 font=("Roboto", 14, "bold"), text_color="#f39c12").pack(pady=(10, 5))

    app.char_scroll = ctk.CTkScrollableFrame(app, height=230)
    app.char_scroll.pack(padx=20, fill="x")
    app.char_widgets = {}

    app.char_placeholder = ctk.CTkLabel(
        app.char_scroll,
        text="Clica em 'ANALISAR LIVRO' ou 'CARREGAR ANÁLISE' para começar...",
        text_color="#666", font=("Roboto", 12)
    )
    app.char_placeholder.pack(pady=40)


def build_audio_controls(app):
    audio_frame = ctk.CTkFrame(app)
    audio_frame.pack(pady=10, padx=20, fill="x")
    ctk.CTkLabel(audio_frame, text="Velocidade Global:").grid(row=0, column=0, padx=10)
    app.speed_slider = ctk.CTkSlider(audio_frame, from_=0.7, to=1.3,
                                      command=app.update_speed_label)
    app.speed_slider.set(1.0)
    app.speed_slider.grid(row=0, column=1, padx=10, sticky="ew")
    app.label_speed_val = ctk.CTkLabel(audio_frame, text="1.00x", width=60)
    app.label_speed_val.grid(row=0, column=2, padx=10)
    audio_frame.grid_columnconfigure(1, weight=1)


def build_progress_section(app):
    app.progress_bar = ctk.CTkProgressBar(app, width=1000)
    app.progress_bar.set(0)
    app.progress_bar.pack(pady=(5, 3), padx=20, fill="x")
    app.label_progress = ctk.CTkLabel(app, text="", font=("Roboto", 11), text_color="#888")
    app.label_progress.pack()
    app.textbox = ctk.CTkTextbox(app, width=1050, height=160, font=("Consolas", 11))
    app.textbox.pack(pady=10, padx=20)
