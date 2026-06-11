# 🔊 Guia da Pasta de Sons — Modo Cinema

## Estrutura recomendada

```
sounds/
├── nature/
│   ├── rain_heavy.wav
│   ├── rain_light.wav
│   ├── thunder.wav
│   ├── wind.wav
│   ├── sea_waves.wav
│   ├── fire_crackling.wav
│   ├── birds.wav
│   └── forest_ambient.wav
├── city/
│   ├── traffic.wav
│   ├── crowd_busy.wav
│   ├── crowd_murmur.wav
│   ├── sirens.wav
│   ├── cafe_ambient.wav
│   └── clock_bell.wav
├── interior/
│   ├── door_open.wav
│   ├── door_close.wav
│   ├── door_knock.wav
│   ├── footsteps_wood.wav
│   ├── footsteps_stone.wav
│   ├── glass_break.wav
│   ├── clock_ticking.wav
│   ├── fire_crackle.wav
│   └── chair_creak.wav
├── action/
│   ├── explosion.wav
│   ├── gunshot.wav
│   ├── sword_clash.wav
│   ├── horse_gallop.wav
│   └── scream.wav
└── music/
    ├── dramatic.wav       ← loops de música de fundo
    ├── tense.wav
    ├── romantic.wav
    ├── sad.wav
    ├── happy.wav
    ├── mystery.wav
    ├── peaceful.wav
    └── epic.wav
```

## Onde obter sons gratuitos

- **Freesound.org** — grande biblioteca CC (requer conta gratuita)
- **ZapSplat.com** — efeitos sonoros gratuitos
- **Pixabay Audio** — sem necessidade de atribuição
- **BBC Sound Effects** — https://sound-effects.bbcrewind.co.uk

## Requisitos técnicos

- Formato: **WAV** (recomendado), MP3, OGG ou FLAC
- Sample rate: qualquer (FFmpeg converte automaticamente para 24kHz)
- Duração: loops de música devem ter pelo menos 10 segundos

## Como o Modo Cinema funciona

1. **Análise:** O Ollama lê cada segmento de texto e identifica sons implícitos
   ("começou a chover" → `rain`, "a porta bateu" → `door_close`)

2. **Correspondência:** A aplicação procura o ficheiro mais próximo na pasta `sounds/`
   usando correspondência exacta → parcial → fuzzy

3. **Mixagem:** FFmpeg combina a voz + efeitos + música com os volumes configurados

4. **Edição manual:** No painel "Eventos Sonoros Detetados" podes:
   - Alterar o nome do som (ex: `rain` → `rain_heavy`)
   - Ajustar posição: `before` / `during` / `after`
   - Ajustar volume com o slider
   - Ver quais sons foram encontrados (✅) ou não (❌) na DB

## Volumes sugeridos

| Elemento   | dB sugerido | Notas                         |
|------------|-------------|-------------------------------|
| Voz        | 0 dB        | referência                    |
| SFX leves  | -8 a -12    | porta, passos, relógio        |
| SFX fortes | -3 a -6     | trovão, explosão              |
| Música     | -18 a -25   | fundo subtil                  |
