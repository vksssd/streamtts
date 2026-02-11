import sounddevice as sd
import numpy as np
import threading
import queue
import sys
import tty
import termios
from piper import PiperVoice

# --- 1. Setup & Model ---
MODEL_PATH = "en_US-amy-medium.onnx"
print("Initializing Voice Engine...")
voice = PiperVoice.load(MODEL_PATH, use_cuda=False)

samplerate = 22050
blocksize = 1024 
audio_queue = queue.Queue(maxsize=100)
text_queue = queue.Queue()

# --- 2. Audio Engine ---
def audio_callback(outdata, frames, time, status):
    try:
        chunk = audio_queue.get_nowait()
        if len(chunk) < frames:
            outdata[:len(chunk)] = chunk
            outdata[len(chunk):] = np.zeros((frames - len(chunk), 1), dtype=np.float32)
        else:
            outdata[:] = chunk[:frames]
    except queue.Empty:
        outdata[:] = np.zeros((frames, 1), dtype=np.float32)

stream = sd.OutputStream(
    samplerate=samplerate, channels=1, blocksize=blocksize,
    callback=audio_callback, dtype=np.float32
)
stream.start()

# --- 3. The Worker (Optimized with your discovered attribute) ---
def tts_worker():
    while True:
        word = text_queue.get()
        if word is None: break
        
        try:
            # We use the generator to get AudioChunk objects
            for chunk_obj in voice.synthesize(word + " "):
                # Using the attribute we found earlier
                audio_float32 = chunk_obj.audio_float_array
                
                # Moderate volume boost
                audio_float32 = audio_float32 * 1.8
                
                for i in range(0, len(audio_float32), blocksize):
                    block = audio_float32[i:i+blocksize].reshape(-1, 1)
                    if len(block) < blocksize:
                        block = np.vstack((block, np.zeros((blocksize - len(block), 1), dtype=np.float32)))
                    audio_queue.put(block)
        except Exception as e:
            pass # Keep worker alive on minor errors
        finally:
            text_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

# --- 4. The Live Keyboard Listener ---
def run_live_session():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    buffer = ""
    
    print("\033[92m[Live]\033[0m Ready! Type and press SPACE to speak. (Ctrl+C to quit)\n")
    
    try:
        # Set terminal to raw mode for instant key-press detection
        tty.setraw(fd)
        while True:
            char = sys.stdin.read(1)
            
            if char == '\x03': # Ctrl+C
                break
            
            # Handle Backspace
            if char in ('\x7f', '\x08'):
                if len(buffer) > 0:
                    buffer = buffer[:-1]
                    sys.stdout.write('\b \b')
                    sys.stdout.flush()
                continue

            # Echo the character to the screen
            sys.stdout.write(char)
            sys.stdout.flush()

            # Trigger speech on Space or Enter
            if char in (" ", "\r", "\n"):
                word_to_speak = buffer.strip()
                if word_to_speak:
                    text_queue.put(word_to_speak)
                buffer = ""
                if char in ("\r", "\n"):
                    sys.stdout.write("\n")
            else:
                buffer += char
    finally:
        # IMPORTANT: Restore terminal settings
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print("\n\nStopping audio stream...")

if __name__ == "__main__":
    run_live_session()
    stream.stop()
    stream.close()