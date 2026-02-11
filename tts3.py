import sounddevice as sd
import numpy as np
import threading
import queue
import sys
import tty
import termios
import time
from piper import PiperVoice

# --- Initialization ---
MODEL_PATH = "en_US-amy-medium.onnx"
voice = PiperVoice.load(MODEL_PATH, use_cuda=False)

samplerate = 22050
blocksize = 1024 
audio_queue = queue.Queue(maxsize=100)
text_queue = queue.Queue()

def audio_callback(outdata, frames, time, status):
    try:
        chunk = audio_queue.get_nowait()
        outdata[:len(chunk)] = chunk
        if len(chunk) < frames:
            outdata[len(chunk):] = np.zeros((frames - len(chunk), 1), dtype=np.float32)
    except queue.Empty:
        outdata[:] = np.zeros((frames, 1), dtype=np.float32)

stream = sd.OutputStream(samplerate=samplerate, channels=1, blocksize=blocksize, callback=audio_callback, dtype=np.float32)
stream.start()

def tts_worker():
    while True:
        phrase = text_queue.get()
        if phrase is None: break
        try:
            # We add a tiny bit of padding to the text for more natural pauses
            for chunk_obj in voice.synthesize(phrase + "  "):
                audio_float32 = chunk_obj.audio_float_array * 1.5
                for i in range(0, len(audio_float32), blocksize):
                    block = audio_float32[i:i+blocksize].reshape(-1, 1)
                    if len(block) < blocksize:
                        block = np.vstack((block, np.zeros((blocksize - len(block), 1), dtype=np.float32)))
                    audio_queue.put(block)
        except Exception: pass
        finally: text_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

# --- The Natural Sentence-Listener ---
def run_live_session():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    
    input_buffer = ""
    last_key_time = time.time()
    IDLE_TIMEOUT = 1.2  # If you stop typing for 1.2s, it speaks.
    
    print("\033[92m[Natural Mode]\033[0m Speaking on punctuation or natural pauses. (Ctrl+C to quit)\n")
    
    try:
        tty.setraw(fd)
        while True:
            # Concept: Check if the user is typing or idling
            # This allows us to trigger speech even without punctuation.
            if sys.stdin in [sys.stdin]:
                # Non-blocking check for key press
                import select
                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                
                if rlist:
                    char = sys.stdin.read(1)
                    if char == '\x03': break # Ctrl+C
                    
                    # Basic echo
                    sys.stdout.write(char)
                    sys.stdout.flush()
                    
                    # Sentence boundary detection
                    if char in (".", "!", "?", "\r", "\n"):
                        if input_buffer.strip():
                            text_queue.put(input_buffer.strip() + char)
                            input_buffer = ""
                    else:
                        input_buffer += char
                    
                    last_key_time = time.time()
                
                # Idle check: If buffer is full and user hasn't typed in 1.2s
                elif input_buffer.strip() and (time.time() - last_key_time > IDLE_TIMEOUT):
                    text_queue.put(input_buffer.strip())
                    input_buffer = ""

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

if __name__ == "__main__":
    run_live_session()