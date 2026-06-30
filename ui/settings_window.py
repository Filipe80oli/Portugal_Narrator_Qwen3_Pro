# ui/settings_window.py
import json
from pathlib import Path
import customtkinter as ctk

SETTINGS_FILE = Path("config/user_settings.json")
DEFAULT_SETTINGS = {
    "max_block_size": 3000,
    "ollama_timeout_small": 120,
    "ollama_timeout_medium": 300,
    "ollama_timeout_large": 600,
    "ollama_warmup_timeout": 300,
    "ollama_temperature": 0.1,
    "ollama_num_predict": 8192,
    "ollama_batch_size": 3,
    "tts_pace_default": 1.0,
    "tts_max_retries": 3,
    "ollama_model": "gemma3:27b",
    "auto_revise": True,
    "test_mode": False          # <-- NOVO
}


def load_settings():
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for k, v in DEFAULT_SETTINGS.items():
                    data.setdefault(k, v)
                return data
        except:
            return DEFAULT_SETTINGS.copy()
    return DEFAULT_SETTINGS.copy()


def save_settings(settings):
    SETTINGS_FILE.parent.mkdir(exist_ok=True)
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)


class SettingsWindow(ctk.CTkToplevel):
    def __init__(self, parent, current_settings, callback_on_save):
        super().__init__(parent)
        self.title("Definições do Sistema")
        self.geometry("520x720")  # Aumentado para caber mais um checkbox
        self.resizable(False, False)
        self.callback = callback_on_save
        self.settings = current_settings.copy()

        # ─── TORNAR JANELA MODAL E À FRENTE ──────────────────────
        self.transient(parent)
        self.grab_set()
        self.focus_force()
        self.attributes('-topmost', True)

        # ─── CONSTRUIR UI ──────────────────────────────────────────
        self._build_ui()

        # ─── QUANDO FECHAR, LIBERTA O GRAB ──────────────────────
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        main_frame = ctk.CTkScrollableFrame(self, width=480, height=660)
        main_frame.pack(fill="both", expand=True, padx=15, pady=15)

        ctk.CTkLabel(main_frame, text="Ajustes Fino da Análise",
                     font=("Roboto", 16, "bold")).pack(pady=(0, 15))

        # ─── Modelo Ollama ──────────────────────────────────────
        self._add_model_selector(main_frame)

        # ─── Bloco ──────────────────────────────────────────────
        self._add_slider(main_frame, "Tamanho máximo do bloco (caracteres):",
                         "max_block_size", 500, 8000, 500)

        # ─── Timeouts ───────────────────────────────────────────
        self._add_slider(main_frame, "Timeout pequeno (segundos):",
                         "ollama_timeout_small", 30, 300, 10)
        self._add_slider(main_frame, "Timeout médio (segundos):",
                         "ollama_timeout_medium", 60, 600, 30)
        self._add_slider(main_frame, "Timeout grande (segundos):",
                         "ollama_timeout_large", 120, 1200, 60)
        self._add_slider(main_frame, "Timeout warmup (segundos):",
                         "ollama_warmup_timeout", 30, 300, 10)

        # ─── Ollama ─────────────────────────────────────────────
        self._add_slider(main_frame, "Temperatura (0.01–0.5):",
                         "ollama_temperature", 1, 50, 1, scale_divide=100)
        self._add_slider(main_frame, "Tokens máximos (num_predict):",
                         "ollama_num_predict", 512, 8192, 256)
        self._add_slider(main_frame, "Batch size (pedidos paralelos):",
                         "ollama_batch_size", 1, 10, 1)

        # ─── TTS ────────────────────────────────────────────────
        self._add_slider(main_frame, "Pace padrão TTS:",
                         "tts_pace_default", 50, 150, 5, scale_divide=100)
        self._add_slider(main_frame, "Máximo de tentativas TTS:",
                         "tts_max_retries", 1, 5, 1)

        # ─── Opções avançadas ────────────────────────────────────
        self._add_checkbox(main_frame, "Revisão automática de falas (Ollama)",
                           "auto_revise")
        self._add_checkbox(main_frame, "Modo de teste (apenas 3 capítulos ou 30k caracteres)",
                           "test_mode")   # <-- NOVO CHECKBOX

        # ─── Botões ─────────────────────────────────────────────
        btn_frame = ctk.CTkFrame(main_frame)
        btn_frame.pack(pady=20)

        ctk.CTkButton(btn_frame, text="Guardar e Fechar",
                      command=self._save, fg_color="#2980b9").pack(side="left", padx=10)
        ctk.CTkButton(btn_frame, text="Cancelar",
                      command=self._on_close, fg_color="#7f8c8d").pack(side="left", padx=10)

    def _add_model_selector(self, parent):
        from core.ollama_analyzer import get_ollama_models
        models = get_ollama_models("http://localhost:11434")
        if not models:
            models = ["gemma3:27b"]

        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", pady=8)

        ctk.CTkLabel(frame, text="Modelo Ollama:", font=("Roboto", 11)).pack(anchor="w")

        current_model = self.settings.get("ollama_model", models[0] if models else "gemma3:27b")
        if current_model not in models:
            current_model = models[0] if models else "gemma3:27b"

        self.model_var = ctk.StringVar(value=current_model)
        self.model_combobox = ctk.CTkComboBox(
            frame,
            values=models,
            variable=self.model_var,
            width=300,
            font=("Roboto", 11)
        )
        self.model_combobox.pack(anchor="w", padx=5, pady=5)

    def _add_slider(self, parent, label_text, key, from_, to_, step, scale_divide=1):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", pady=5)

        label = ctk.CTkLabel(frame, text=label_text, font=("Roboto", 11))
        label.pack(anchor="w")

        value = self.settings.get(key, DEFAULT_SETTINGS[key])
        display_value = value * scale_divide if scale_divide != 1 else value

        if "temperature" in key:
            number_of_steps = 49
        else:
            number_of_steps = None

        slider = ctk.CTkSlider(
            frame,
            from_=from_,
            to=to_,
            number_of_steps=number_of_steps,
            command=lambda v, k=key, d=scale_divide: self._update_label(k, v, d)
        )
        slider.set(display_value)
        slider.pack(fill="x", padx=5)

        if "temperature" in key:
            initial_text = f"{display_value / scale_divide:.2f}" if scale_divide != 1 else str(display_value)
        else:
            initial_text = str(display_value) if scale_divide == 1 else f"{display_value / scale_divide:.2f}"

        val_label = ctk.CTkLabel(frame, text=initial_text, width=60)
        val_label.pack(anchor="e")

        setattr(self, f"_slider_{key}", slider)
        setattr(self, f"_label_{key}", val_label)
        self.settings[key] = value

    def _add_checkbox(self, parent, label_text, key):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", pady=5)

        var = ctk.BooleanVar(value=self.settings.get(key, DEFAULT_SETTINGS.get(key, True)))
        cb = ctk.CTkCheckBox(
            frame,
            text=label_text,
            variable=var,
            font=("Roboto", 11),
            command=lambda: self.settings.__setitem__(key, var.get())
        )
        cb.pack(anchor="w", padx=5, pady=2)
        setattr(self, f"_check_{key}", cb)

    def _update_label(self, key, raw_value, scale_divide):
        if "temperature" in key:
            val = raw_value / scale_divide
            display = f"{val:.2f}"
        else:
            if scale_divide == 1:
                display = str(int(raw_value))
            else:
                val = raw_value / scale_divide
                display = f"{val:.2f}"

        getattr(self, f"_label_{key}").configure(text=display)
        if "temperature" in key:
            self.settings[key] = val
        else:
            self.settings[key] = int(raw_value) if scale_divide == 1 else val

    def _save(self):
        for key in DEFAULT_SETTINGS.keys():
            slider = getattr(self, f"_slider_{key}", None)
            if slider is not None:
                raw = slider.get()
                if "temperature" in key or "pace" in key:
                    self.settings[key] = raw / 100.0
                else:
                    self.settings[key] = int(raw)
            checkbox = getattr(self, f"_check_{key}", None)
            if checkbox is not None:
                self.settings[key] = checkbox.get()

        if hasattr(self, 'model_var'):
            self.settings["ollama_model"] = self.model_var.get()

        save_settings(self.settings)
        if self.callback:
            self.callback(self.settings)
        self._on_close()

    def _on_close(self):
        self.grab_release()
        self.attributes('-topmost', False)
        self.destroy()