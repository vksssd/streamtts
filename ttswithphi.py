import sounddevice as sd
import numpy as np
import threading
import queue
import sys
import tty
import termios
import time
from piper import PiperVoice

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch


# --- 1. Global Setup ---
MODEL_PATH = "en_US-amy-medium.onnx"
voice = PiperVoice.load(MODEL_PATH, use_cuda=False)

samplerate = 22050
blocksize = 1024 
audio_queue = queue.Queue(maxsize=100)
text_queue = queue.Queue()

# --- 2. Audio Output Engine ---
def audio_callback(outdata, frames, time, status):
    try:
        chunk = audio_queue.get_nowait()
        outdata[:len(chunk)] = chunk
        if len(chunk) < frames:
            outdata[len(chunk):] = 0
    except queue.Empty:
        outdata[:] = 0

stream = sd.OutputStream(samplerate=samplerate, channels=1, blocksize=blocksize, callback=audio_callback, dtype=np.float32)
stream.start()

# --- Tiny LLM Setup ---
# --- Phi Setup ---
# LLM_MODEL = "microsoft/phi-1_5"
LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.float32,   # safer for Pi CPU
)

model.eval()

print("Phi ready!")


def stream_llm_response(user_input):
    print("Generating... please wait.")
    start = time.time()

    prompt = f"""
You are a helpful assistant.
Answer clearly and concisely.

Question: {user_input}
Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.2,
            top_k=40,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    end = time.time()
    print(f"\nGeneration took {end - start:.2f} seconds")
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt part
    answer = full_text.split("Answer:")[-1].strip()

    print(answer)

    # Send sentence by sentence to TTS
    sentences = answer.replace("?", ".").replace("!", ".").split(".")
    for s in sentences:
        s = s.strip()
        if s:
            text_queue.put(s)


# --- 3. The Voice Worker (Consumer) ---
def tts_worker():
    while True:
        phrase = text_queue.get()
        if phrase is None: break
        try:
            # We add a space to help Piper with word boundaries
            for chunk_obj in voice.synthesize(phrase + " "):
                audio_float32 = chunk_obj.audio_float_array * 1.5
                for i in range(0, len(audio_float32), blocksize):
                    block = audio_float32[i:i+blocksize].reshape(-1, 1)
                    if len(block) < blocksize:
                        block = np.vstack((block, np.zeros((blocksize - len(block), 1), dtype=np.float32)))
                    audio_queue.put(block)
        except Exception: pass
        finally: text_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

# --- 4. The Keyboard Streamer (Producer) ---
def run_live_stream():
    print("\033[93m[LLM STREAM MODE]\033[0m Type a question and press ENTER. (Ctrl+C to quit)\n")
    
    try:
        while True:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue

            print("LLM:", end=" ", flush=True)
            stream_llm_response(user_input)

    except KeyboardInterrupt:
        print("\nStream ended.")


if __name__ == "__main__":
    run_live_stream()

