import sounddevice as sd
import numpy as np
import threading
import queue
import time
from piper import PiperVoice

# --- 1. The Global Audio Pipe ---
# We use a queue to store 1024-sample blocks
audio_queue = queue.Queue(maxsize=100)
samplerate = 22050
blocksize = 1024

def audio_callback(outdata, frames, time, status):
    """
    This function is called by the sound card whenever it needs more sound.
    """
    try:
        # Get one block of audio (1024 samples)
        chunk = audio_queue.get_nowait()
        
        # Check if the chunk matches the requested frames (usually 1024)
        if len(chunk) == frames:
            outdata[:] = chunk
        else:
            # If the chunk is smaller (the very last bit of speech), 
            # fill what we can and zero out the rest.
            outdata[:len(chunk)] = chunk
            outdata[len(chunk):] = 0
    except queue.Empty:
        # If no audio is ready, play silence
        outdata[:] = 0

# Start the stream immediately
stream = sd.OutputStream(
    samplerate=samplerate, 
    channels=1, 
    blocksize=blocksize, 
    callback=audio_callback, 
    dtype=np.float32
)
stream.start()

# --- 2. The Synthesis Engine ---
def speak_text(text, voice_model):
    # Split by punctuation for natural sentence-level prosody
    import re
    sentences = re.split(r'(?<=[.!?]) +', text)
    
    for sentence in sentences:
        if not sentence.strip(): continue
        
        print(f"Synthesizing: {sentence}")
        
        # Piper generator provides AudioChunk objects
        for chunk_obj in voice_model.synthesize(sentence):
            # Use the float array we identified earlier
            raw_audio = chunk_obj.audio_float_array * 1.5
            
            # Break the raw audio into exactly 1024-sample blocks
            for i in range(0, len(raw_audio), blocksize):
                block = raw_audio[i:i+blocksize].reshape(-1, 1)
                
                # If the last piece of the sentence is < 1024, pad it with silence
                if len(block) < blocksize:
                    padding = np.zeros((blocksize - len(block), 1), dtype=np.float32)
                    block = np.vstack((block, padding))
                
                audio_queue.put(block)

# --- 3. Run it ---
if __name__ == "__main__":
    MODEL_PATH = "en_US-amy-medium.onnx"
    voice = PiperVoice.load(MODEL_PATH, use_cuda=False)

    text = (
        "Streaming is now stable. By padding the final block of each sentence, "
        "we prevent the broadcast error. This allows for a continuous flow of "
        "speech without the robotic clankiness of word by word synthesis."
    )

    # Run synthesis in a separate thread so it doesn't block the main program
    synthesis_thread = threading.Thread(target=speak_text, args=(text, voice))
    synthesis_thread.start()

    # Wait for everything to finish
    synthesis_thread.join()
    while not audio_queue.empty():
        time.sleep(0.1)
    
    time.sleep(0.5)
    stream.stop()
    print("Stream finished successfully.")