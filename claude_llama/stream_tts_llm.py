import sounddevice as sd
import numpy as np
import threading
import queue
from piper import PiperVoice
from llama_cpp import Llama

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

stream = sd.OutputStream(
    samplerate=samplerate, 
    channels=1, 
    blocksize=blocksize, 
    callback=audio_callback, 
    dtype=np.float32
)
stream.start()

# --- 3. LLM Setup with llama.cpp ---
# Download a GGUF model from HuggingFace, examples:
# - Llama 3.2 3B Instruct Q4: https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF
# - Phi-3.5 Mini Instruct Q4: https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF
# - Qwen2.5 3B Instruct Q4: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF

LLM_MODEL_PATH = "./models/llama-3.2-3b-instruct-q4_k_m.gguf"  # Update this path

print("Loading LLM with llama.cpp...")
llm = Llama(
    model_path=LLM_MODEL_PATH,
    n_ctx=2048,           # Context window
    n_threads=4,          # CPU threads (adjust based on your CPU)
    n_gpu_layers=0,       # Set to 35+ if you have GPU with CUDA
    verbose=False
)
print("LLM ready!")

def stream_llm_response(user_input):
    """Stream LLM response token by token"""
    
    prompt = f"""You are a helpful assistant. Answer clearly and concisely.

Question: {user_input}
Answer:"""

    buffer = ""

    # Stream tokens from llama.cpp
    for chunk in llm(
        prompt,
        max_tokens=140,
        temperature=0.3,
        top_p=0.9,
        stream=True,
        stop=["Question:", "\n\n"]  # Stop sequences
    ):
        # Extract text from chunk
        token_text = chunk['choices'][0]['text']
        
        print(token_text, end="", flush=True)
        buffer += token_text

        # Speak on sentence boundary
        if any(p in token_text for p in ".!?"):
            text_queue.put(buffer.strip())
            buffer = ""

    # Send remaining buffer
    if buffer.strip():
        text_queue.put(buffer.strip())
    
    print()  # Newline after completion


# --- 4. The Voice Worker (Consumer) ---
def tts_worker():
    """Background thread that converts text to speech"""
    while True:
        phrase = text_queue.get()
        if phrase is None:
            break
        try:
            # Synthesize speech with Piper
            for chunk_obj in voice.synthesize(phrase + " "):
                audio_float32 = chunk_obj.audio_float_array * 1.5
                
                # Split into blocks for streaming
                for i in range(0, len(audio_float32), blocksize):
                    block = audio_float32[i:i+blocksize].reshape(-1, 1)
                    if len(block) < blocksize:
                        block = np.vstack((
                            block, 
                            np.zeros((blocksize - len(block), 1), dtype=np.float32)
                        ))
                    audio_queue.put(block)
        except Exception as e:
            print(f"\nTTS error: {e}")
        finally:
            text_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()


# --- 5. Main Interactive Loop ---
def run_live_stream():
    """Main loop for user interaction"""
    print("\033[93m[LLM STREAM MODE with llama.cpp]\033[0m")
    print("Type a question and press ENTER. (Ctrl+C to quit)\n")
    
    try:
        while True:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue

            print("LLM: ", end="", flush=True)
            stream_llm_response(user_input)

    except KeyboardInterrupt:
        print("\n\nStream ended. Goodbye!")
    finally:
        stream.stop()
        stream.close()


if __name__ == "__main__":
    run_live_stream()