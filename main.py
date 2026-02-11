from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
import numpy as np
import asyncio
import json

print("🔥 main.py LOADED")

app = FastAPI()
print("🔥 FastAPI app created")

# Use small/medium depending on CPU strength
model = WhisperModel(
    "tiny.en",
    device="cpu",
    compute_type="int8",
)
print("🔥 Whisper model loaded")

SAMPLE_RATE = 16000
SILENCE_SEC = 1.0  # silence duration to trigger transcription
MIN_AUDIO_SEC = 0.5  # minimum speech before transcribing


@app.websocket("/ws/transcribe")
async def transcribe(ws: WebSocket):
    print("🟢 WebSocket connection attempt")
    await ws.accept()
    print("🟢 WebSocket accepted")

    loop = asyncio.get_running_loop()

    audio_buffer = np.zeros(0, dtype=np.float32)
    silence_buffer = 0
    last_sent_text = ""

    try:
        while True:
            msg = await ws.receive()

            # ---- STOP SIGNAL ----
            if msg["type"] == "websocket.receive" and "text" in msg:
                data = json.loads(msg["text"])
                if data.get("event") == "stop":
                    print("🛑 stop received")
                    break

            # ---- AUDIO ----
            if "bytes" in msg:
                chunk = (
                    np.frombuffer(msg["bytes"], dtype=np.int16).astype(np.float32)
                    / 32768.0
                )

                if len(chunk) == 0:
                    continue

                audio_buffer = np.concatenate([audio_buffer, chunk])

                # ---- Simple silence detection (energy based) ----
                energy = np.abs(chunk).mean()

                if energy < 0.01:
                    silence_buffer += len(chunk)
                else:
                    silence_buffer = 0

                silence_duration = silence_buffer / SAMPLE_RATE
                total_audio_duration = len(audio_buffer) / SAMPLE_RATE

                # ---- Trigger transcription on silence ----
                if (
                    silence_duration >= SILENCE_SEC
                    and total_audio_duration >= MIN_AUDIO_SEC
                ):
                    print("🧠 Speech ended → running whisper")

                    segments, _ = await loop.run_in_executor(
                        None,
                        lambda: model.transcribe(
                            audio_buffer,
                            language="en",
                            vad_filter=True,
                        ),
                    )

                    text = " ".join(seg.text for seg in segments).strip()

                    if text and text != last_sent_text:
                        last_sent_text = text
                        print("📝 final:", text)
                        await ws.send_text(text)

                    # Reset buffers after processing
                    audio_buffer = np.zeros(0, dtype=np.float32)
                    silence_buffer = 0

    except WebSocketDisconnect:
        print("🔴 client disconnected")

    # ---- Final flush if remaining audio ----
    if len(audio_buffer) > 0:
        print("🧠 Final flush")

        segments, _ = await loop.run_in_executor(
            None,
            lambda: model.transcribe(
                audio_buffer,
                language="en",
                vad_filter=True,
            ),
        )

        text = " ".join(seg.text for seg in segments).strip()

        try:
            if text:
                await ws.send_text(text)
        except RuntimeError:
            pass

    try:
        await ws.close()
    except:
        pass

    print("🔵 WebSocket closed cleanly")
