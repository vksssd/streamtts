import subprocess
import threading
import queue
import numpy as np
import sounddevice as sd
import signal
import sys
from piper import PiperVoice

# ==========================================
# CONFIG
# ==========================================
LLAMA_BIN = "./llama.cpp/main"
MODEL_PATH = "./llama.cpp/models/tinyllama.gguf"  # change later
PIPER_MODEL = "en_US-amy-medium.onnx"

SAMPLERATE = 22050
BLOCKSIZE = 1024

# ==========================================
# Load Piper
# ==========================================
print("Loading Piper...")
voice = PiperVoice.load(PIPER_MODEL, use_cuda=False)
print("Piper ready!")

audio_queue = queue.Queue(maxsize=100)
text_queue = queue.Queue()

# ==========================================
# Audio Output
# ==========================================
def audio_callback(outdata, frames, time, status):
    try:
        chunk = audio_queue.get_nowait()
        outdata[:len(chunk)] = chunk
        if len(chunk) < frames:
            outdata[len(chunk):] = 0
    except queue.Empty:
        outdata[:] = 0

stream = sd.OutputStream(
    samplerate=SAMPLERATE,
    channels=1,
    blocksize=BLOCKSIZE,
    callback=audio_callback,
    dtype=np.float32
)
stream.start()

# ==========================================
# TTS Worker
# ==========================================
def tts_worker():
    while True:
        phrase = text_queue.get()
        if phrase is None:
            break
        try:
            for chunk_obj in voice.synthesize(phrase + " "):
                audio = chunk_obj.audio_float_array

                for i in range(0, len(audio), BLOCKSIZE):
                    block = audio[i:i+BLOCKSIZE].reshape(-1, 1)
                    if len(block) < BLOCKSIZE:
                        block = np.vstack(
                            (block,
                             np.zeros((BLOCKSIZE - len(block), 1),
                                      dtype=np.float32))
                        )
                    audio_queue.put(block)
        except Exception:
            pass
        finally:
            text_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

# ==========================================
# Llama Streaming
# ==========================================
def stream_llama(prompt):

    cmd = [
        LLAMA_BIN,
        "-m", MODEL_PATH,
        "-p", f"<|user|>\n{prompt}\n<|assistant|>",
        "-n", "200",
        "--temp", "0.6",
        "--top-p", "0.9",
        "--repeat_penalty", "1.1",
        "--ctx-size", "2048",
        "--color",
        "--no-display-prompt"
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1
    )

    buffer = ""

    print("\nAssistant:", end=" ", flush=True)

    for line in process.stdout:
        print(line, end="", flush=True)
        buffer += line

        # Speak when sentence ends
        if any(p in line for p in ".!?"):
            text_queue.put(buffer.strip())
            buffer = ""

    if buffer.strip():
        text_queue.put(buffer.strip())

    process.wait()

# ==========================================
# Clean Shutdown
# ==========================================
def shutdown(sig=None, frame=None):
    print("\nShutting down...")
    text_queue.put(None)
    stream.stop()
    stream.close()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)

# ==========================================
# Main Loop
# ==========================================
print("\n=== Llama Voice Assistant ===")

while True:
    user_input = input("\nYou: ").strip()
    if not user_input:
        continue

    stream_llama(user_input)
