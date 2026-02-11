import sounddevice as sd
import numpy as np
import threading
import queue
import sys
import tty
import termios
from piper import PiperVoice

# --- Initialization (Same as before) ---
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
            for chunk_obj in voice.synthesize(phrase):
                audio_float32 = chunk_obj.audio_float_array * 1.8
                for i in range(0, len(audio_float32), blocksize):
                    block = audio_float32[i:i+blocksize].reshape(-1, 1)
                    if len(block) < blocksize:
                        block = np.vstack((block, np.zeros((blocksize - len(block), 1), dtype=np.float32)))
                    audio_queue.put(block)
        except Exception: pass
        finally: text_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

# --- The "Natural Flow" Listener ---
def run_live_session():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    
    input_buffer = ""      # Holds current characters
    word_window = []       # Holds the 3-4 words context
    WINDOW_SIZE = 3        # Number of words to wait for
    
    print("\033[95m[Natural Mode]\033[0m Waiting for 3 words for better inflection...")
    
    try:
        tty.setraw(fd)
        while True:
            char = sys.stdin.read(1)
            if char == '\x03': break # Ctrl+C
            
            # Standard echo and backspace logic
            if char in ('\x7f', '\x08'):
                if len(input_buffer) > 0:
                    input_buffer = input_buffer[:-1]
                    sys.stdout.write('\b \b')
                    sys.stdout.flush()
                continue
            
            sys.stdout.write(char)
            sys.stdout.flush()

            # Word boundary detected
            if char in (" ", "\r", "\n", ".", ",", "!", "?"):
                word = input_buffer.strip()
                if word:
                    word_window.append(word)
                input_buffer = ""

                # Trigger logic: 
                # 1. We reached the window size
                # 2. OR the user typed punctuation
                # 3. OR the user pressed Enter
                if len(word_window) >= WINDOW_SIZE or char in (".", "!", "?", "\r", "\n"):
                    if word_window:
                        phrase = " ".join(word_window)
                        text_queue.put(phrase)
                        word_window = [] # Clear window after sending
                
                if char in ("\r", "\n"):
                    sys.stdout.write("\n")
            else:
                input_buffer += char
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

if __name__ == "__main__":
    run_live_session()