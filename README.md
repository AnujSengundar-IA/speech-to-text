# Speech-to-Text Streaming (Whisper + Parakeet + IndicConformer)

Real-time speech-to-text server and browser client using FastAPI WebSockets.

- Whisper endpoint: `/ws/transcribe`
- Parakeet endpoint: `/ws/transcribe/parakeet`
- IndicConformer endpoint: `/ws/transcribe/indicconformer`
- Browser demo: `index.html` + `pcm-worklet.js`

## Features

- Real-time streaming transcription from microphone audio
- Partial and final transcript events
- Three model backends:
  - Faster-Whisper (`small.en` by default)
  - NVIDIA Parakeet (`nvidia/parakeet-ctc-0.6b`)
  - AI4Bharat IndicConformer (NeMo `.nemo` checkpoint, 22 Indian languages)
- Adaptive silence detection and automatic segment finalization
- Optional WebSocket origin allowlist and token auth
- Bounded buffering and backpressure control

## Project Structure

```text
main.py                      # thin entrypoint, exports app
stt_service/
  application.py             # FastAPI app, lifespan, websocket routes
  config.py                  # environment config
  transcribers.py            # Whisper + Parakeet + IndicConformer backends
  stream.py                  # streaming websocket loop + emit logic
  session.py                 # session audio/chunk state
  ws_utils.py                # websocket helpers
  security.py                # origin/token checks
index.html                   # browser demo UI
pcm-worklet.js               # AudioWorklet streaming PCM16 to websocket
.env.example                 # environment template
```

## Requirements

- Python `3.12+`
- A modern browser with AudioWorklet support
- For Parakeet on GPU: CUDA-enabled PyTorch build

## Setup

### Option A: `uv` (recommended)

```bash
uv sync
cp .env.example .env
```

### Option B: `venv` + `pip`

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
cp .env.example .env
```

## Run the API server

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8999 --reload
```

or with venv:

```bash
source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8999 --reload
```

## Run the browser client

Serve the project root as static files (any simple HTTP server works):

```bash
python -m http.server 8080
```

Open `http://localhost:8080/index.html`.

1. Choose model (`Whisper`, `Parakeet`, or `IndicConformer`) in the dropdown.
2. For IndicConformer, pick a source language.
3. Click **Start**.
4. Speak.
5. Click **Stop** to flush a final segment.

## WebSocket APIs

### 1) Whisper

- URL: `ws://localhost:8999/ws/transcribe`

### 2) Parakeet

- URL: `ws://localhost:8999/ws/transcribe/parakeet`

### 3) IndicConformer

- URL: `ws://localhost:8999/ws/transcribe/indicconformer?language_id=hi`
- `language_id` is optional. If omitted, server uses `INDICCONFORMER_DEFAULT_LANGUAGE_ID`.

All endpoints accept the same protocol.

### Client -> Server

- Binary frames: 16-bit PCM mono audio (`int16`, sample rate from `SAMPLE_RATE`, default `16000`)
- Text JSON control messages:

```json
{"event":"ping"}
```

```json
{"event":"stop"}
```

### Server -> Client

All responses are JSON with `type` field.

- `{"type":"status","status":"connected"}`
- `{"type":"status","status":"pong"}`
- `{"type":"status","status":"busy","reason":"backpressure"}`
- `{"type":"status","status":"buffer_trimmed","removed_sec":1.23}`
- `{"type":"partial","text":"...","end":2.10,"reason":"interval","window_start_sec":0.00}`
- `{"type":"final","text":"...","end":3.40,"reason":"silence|stop|max_segment|final_flush","window_start_sec":1.00}`
- `{"type":"error","message":"client_timeout|transcription_failed|internal_error|model_unavailable"}`

## Configuration

Copy `.env.example` to `.env` and adjust as needed.

### Whisper

- `WHISPER_MODEL_SIZE` (default `small.en`)
- `WHISPER_MODEL_DEVICE` (`cpu`, `cuda`, `cuda:0`)
- `WHISPER_COMPUTE_TYPE` (`int8`, `float16`, ...)
- `WHISPER_LANGUAGE` (default `en`)
- `WHISPER_VAD_FILTER` (`true`/`false`)

### Parakeet

- `PARAKEET_MODEL_ID` (default `nvidia/parakeet-ctc-0.6b`)
- `PARAKEET_DEVICE` (`cpu`, `cuda`, `cuda:0`)
- `PARAKEET_TORCH_DTYPE` (`float32`, `float16`, `bfloat16`)
- `nvidia/parakeet-tdt-*` checkpoints are NeMo archives and are not loadable via the transformers backend used by this project.

### IndicConformer

- default model source: `ai4bharat/indic-conformer-600m-multilingual`
- `INDICCONFORMER_CHECKPOINT_PATH` (optional override; one of:)
- local `.nemo` checkpoint path
- local HF snapshot directory path
- HF repo id (for example `ai4bharat/indic-conformer-600m-multilingual`)
- `INDICCONFORMER_DEVICE` (`cpu`, `cuda`, `cuda:0`)
- `INDICCONFORMER_DECODER` (`ctc` or `rnnt`)
- `INDICCONFORMER_DEFAULT_LANGUAGE_ID` (default `hi`)
- `INDICCONFORMER_BATCH_SIZE` (default `1`)

22-language codes exposed in the demo UI:
`as`, `bn`, `brx`, `doi`, `gu`, `hi`, `kn`, `ks`, `kok`, `mai`, `ml`, `mni`, `mr`, `ne`, `or`, `pa`, `sa`, `sat`, `sd`, `ta`, `te`, `ur`.

### Audio / segmentation

- `SAMPLE_RATE` (default `16000`)
- `MIN_AUDIO_SEC` (minimum audio before transcript)
- `SILENCE_SEC` (silence duration to finalize)
- `MAX_SEGMENT_SEC` (hard max segment duration)
- `MAX_BUFFER_SEC` (max in-memory audio buffer)
- `PARTIAL_INTERVAL_SEC` (how often partials are emitted)
- `OVERLAP_SEC` (audio overlap between transcribes)

### Silence detection

- `SILENCE_MULTIPLIER`
- `SILENCE_RMS_FLOOR`
- `NOISE_ALPHA`
- `MIN_NOISE_FLOOR`

### Connection / throughput

- `CLIENT_TIMEOUT_SEC`
- `MAX_CONCURRENT_TRANSCRIBES`
- `TRANSCRIBE_ACQUIRE_TIMEOUT_SEC`

### Security (optional)

- `WS_ALLOWED_ORIGINS` (comma-separated)
- `WS_AUTH_TOKEN`

If `WS_AUTH_TOKEN` is set, clients must provide token using query param `?token=...` or header `x-api-key`.

## Notes on model loading

- Parakeet and IndicConformer model loading happens at app startup.
- If a model load fails, server still runs and other endpoints still work.
- Unavailable endpoint then returns:

```json
{"type":"error","message":"model_unavailable"}
```

## Troubleshooting

### `pyenv: version '3.12' is not installed`

Install Python 3.12, or adjust `.python-version` and your virtual environment to a compatible version (`>=3.12`).

### Browser says WebSocket closed immediately

Check server logs for:

- origin rejected (`WS_ALLOWED_ORIGINS`)
- missing/invalid token (`WS_AUTH_TOKEN`)

### No transcripts

- Ensure browser is sending mono PCM16 at `16000` Hz (default client already does)
- Confirm model loaded successfully in server startup logs
- Try longer speech segments and reduce `MIN_AUDIO_SEC`

## Development

- Entry app import: `main:app`
- Hot reload command:

```bash
uv run uvicorn main:app --reload --port 8999
```
