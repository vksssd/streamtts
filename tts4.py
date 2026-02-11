import sounddevice as sd
import numpy as np
import threading
import queue
import sys
import tty
import termios
from piper import PiperVoice

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
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    
    current_word = ""
    # How many words to group before speaking (lower = faster, higher = natural)
    CHUNK_SIZE = 2 
    phrase_buffer = []
    
    print("\033[93m[STREAMING ACTIVE]\033[0m Type naturally. Voice will follow. (Ctrl+C to quit)\n")
    
    try:
        tty.setraw(fd)
        while True:
            char = sys.stdin.read(1)
            
            if char == '\x03': # Ctrl+C
                break
                
            # Handle Backspace
            if char in ('\x7f', '\x08'):
                if len(current_word) > 0:
                    current_word = current_word[:-1]
                    sys.stdout.write('\b \b')
                    sys.stdout.flush()
                continue

            # Echo the character to the terminal
            sys.stdout.write(char)
            sys.stdout.flush()

            # If space or punctuation, we have a complete word
            if char in (" ", ".", "!", "?", "\r", "\n"):
                word = current_word.strip()
                if word:
                    phrase_buffer.append(word)
                current_word = ""

                # Trigger speech every 2 words OR on punctuation
                if len(phrase_buffer) >= CHUNK_SIZE or char in (".", "!", "?", "\r", "\n"):
                    if phrase_buffer:
                        text_queue.put(" ".join(phrase_buffer))
                        phrase_buffer = []
                
                if char in ("\r", "\n"):
                    sys.stdout.write("\n")
            else:
                current_word += char
    finally:
        # Crucial: Reset the terminal to normal mode
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print("\nStream ended.")

if __name__ == "__main__":
    run_live_stream()