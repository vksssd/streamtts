import sounddevice as sd
import numpy as np
import threading
import queue
import sys
from piper import PiperVoice
from llama_cpp import Llama  # Replaces transformers/torch

# --- 1. Global Setup ---
# Piper Voice Model
VOICE_MODEL_PATH = "en_US-amy-medium.onnx"
voice = PiperVoice.load(VOICE_MODEL_PATH, use_cuda=False)

# LLM Model (Must be a .gguf file)
LLM_MODEL_PATH = "model.gguf"  # <--- Make sure this file exists!

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

# --- 3. Llama.cpp Setup ---
print(f"Loading Llama model from {LLM_MODEL_PATH}...")
try:
    llm = Llama(
        model_path=LLM_MODEL_PATH,
        n_ctx=2048,        # Context window size
        n_threads=4,       # Number of CPU threads to use
        verbose=False      # Set to True to see C++ logs
    )
    print("Llama ready!")
except Exception as e:
    print(f"\nError loading model: {e}")
    print(f"Make sure '{LLM_MODEL_PATH}' exists in this folder.")
    sys.exit(1)

def stream_llm_response(user_input):
    # Prompt formatting (Adjust based on your specific model, e.g., ChatML or Alpaca)
    prompt = f"""You are a helpful assistant. Answer clearly and concisely.

Question: {user_input}
Answer:"""

    # Stream the generation
    stream = llm(
        prompt,
        max_tokens=140,
        stop=["Question:", "\n\n"], # Stop generation if model tries to start a new Q
        stream=True,                # Enable streaming
        temperature=0.3
    )

    buffer = ""

    for output in stream:
        # Extract the token text
        token = output['choices'][0]['text']
        
        print(token, end="", flush=True)
        buffer += token

        # Speak on sentence boundary
        if any(p in token for p in ".!?"):
            if buffer.strip():
                text_queue.put(buffer.strip())
            buffer = ""

    # Flush remaining text
    if buffer.strip():
        text_queue.put(buffer.strip())


# --- 4. The Voice Worker (Consumer) ---
def tts_worker():
    while True:
        phrase = text_queue.get()
        if phrase is None: break
        try:
            # We add a space to help Piper with word boundaries
            # Note: Piper synthesis can be blocking, so this runs in its own thread
            for chunk_obj in voice.synthesize(phrase + " "):
                audio_float32 = chunk_obj.audio_float_array * 1.5 # Volume boost
                for i in range(0, len(audio_float32), blocksize):
                    block = audio_float32[i:i+blocksize].reshape(-1, 1)
                    if len(block) < blocksize:
                        block = np.vstack((block, np.zeros((blocksize - len(block), 1), dtype=np.float32)))
                    audio_queue.put(block)
        except Exception as e:
            print(f"TTS Error: {e}")
        finally: 
            text_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

# --- 5. The Keyboard Streamer (Producer) ---
def run_live_stream():
    print("\033[93m[LLAMA.CPP STREAM MODE]\033[0m Type a question and press ENTER. (Ctrl+C to quit)\n")
    
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