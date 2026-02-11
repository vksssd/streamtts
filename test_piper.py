import sounddevice as sd
import numpy as np
import threading
import queue
import time
from piper import PiperVoice

# --- 1. Initialization ---
MODEL_PATH = "en_US-amy-medium.onnx"
print("Loading model...")
voice = PiperVoice.load(MODEL_PATH, use_cuda=False)

samplerate = 22050
blocksize = 1024 
audio_queue = queue.Queue()
text_queue = queue.Queue()

# --- 2. Audio Engine ---
def audio_callback(outdata, frames, time, status):
    try:
        chunk = audio_queue.get_nowait()
        outdata[:] = chunk
    except queue.Empty:
        outdata[:] = np.zeros((frames, 1), dtype=np.float32)

print(f"Using Audio Device: {sd.query_devices(kind='output')['name']}")

stream = sd.OutputStream(
    samplerate=samplerate, channels=1, blocksize=blocksize,
    callback=audio_callback, dtype=np.float32
)
stream.start()

# --- 3. The Worker (Using audio_float_array) ---
def tts_worker():
    while True:
        text = text_queue.get()
        if text is None: break
        
        try:
            print(f"[Working] Synthesizing: {text}")
            
            for chunk_obj in voice.synthesize(text):
                # Concept: Use the pre-converted float array for maximum speed
                # This attribute was found via our previous 'dir()' inspection
                audio_float32 = chunk_obj.audio_float_array
                
                # Volume boost (MacBook speakers can be quiet)
                audio_float32 = audio_float32 * 2.0
                
                # Slicing into blocks for sounddevice
                for i in range(0, len(audio_float32), blocksize):
                    block = audio_float32[i:i+blocksize].reshape(-1, 1)
                    if len(block) < blocksize:
                        block = np.vstack((block, np.zeros((blocksize - len(block), 1), dtype=np.float32)))
                    audio_queue.put(block)
            
            print("[Success] Audio added to queue.")
                
        except Exception as e:
            print(f"Inference Error: {e}")
        finally:
            text_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

# --- 4. The Test ---
if __name__ == "__main__":
    test_text = "I am speaking using the floating point array. This is the most efficient method."
    text_queue.put(test_text)
    
    while not audio_queue.empty() or not text_queue.empty():
        time.sleep(0.1)
    
    time.sleep(1.5)
    print("[Done]")
    stream.stop()