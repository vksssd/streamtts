import sounddevice as sd
import numpy as np
import threading
import queue
import sys
import tty
import termios
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
# LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# LLM_MODEL = "facebook/opt-125m"
# LLM_MODEL = "facebook/opt-35m"
# LLM_MODEL = "distilgpt2"
LLM_MODEL = "gpt2"



print("Loading TinyLLM...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.float32,
    device_map="auto"
)
print("TinyLLM ready!")

def stream_llm_response(user_input):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    prompt = f"""
You are a helpful assistant.
Answer clearly and concisely.

Question: {user_input}
Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        min_new_tokens=1,
        max_new_tokens=140,
        temperature=0.3,
        do_sample=True,
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    buffer = ""

    for new_text in streamer:
        print(new_text, end="", flush=True)
        buffer += new_text

        # Speak on sentence boundary
        if any(p in new_text for p in ".!?"):
            text_queue.put(buffer.strip())
            buffer = ""

    if buffer.strip():
        text_queue.put(buffer.strip())



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

