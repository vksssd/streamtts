import sounddevice as sd
import numpy as np
import threading
import queue
import time
from piper import PiperVoice
from llama_cpp import Llama
from typing import List, Dict
import os
import whisper
from collections import deque

# --- Configuration ---
class Config:
    # Paths
    TTS_MODEL_PATH = "en_US-amy-medium.onnx"
    LLM_MODEL_PATH = "./models/llama-3.2-3b-instruct-q4_k_m.gguf"
    WHISPER_MODEL = "base"  # tiny, base, small, medium, large
    
    # Audio settings
    SAMPLE_RATE = 22050
    BLOCK_SIZE = 1024
    AUDIO_QUEUE_SIZE = 100
    VOLUME_MULTIPLIER = 1.5
    
    # Voice Activity Detection (VAD)
    VAD_SAMPLE_RATE = 16000
    VAD_THRESHOLD = 0.02  # Adjust based on environment
    SILENCE_DURATION = 1.5  # Seconds of silence to trigger processing
    MIN_SPEECH_DURATION = 0.5  # Minimum seconds of speech
    
    # LLM settings
    CONTEXT_WINDOW = 2048
    MAX_TOKENS = 200
    TEMPERATURE = 0.7
    TOP_P = 0.9
    CPU_THREADS = 4
    GPU_LAYERS = 0
    
    # Conversation settings
    MAX_HISTORY = 10
    SYSTEM_PROMPT = "You are a helpful, friendly voice assistant. Keep responses concise and natural."


# --- Global State ---
class State:
    def __init__(self):
        self.audio_queue = queue.Queue(maxsize=Config.AUDIO_QUEUE_SIZE)
        self.text_queue = queue.Queue()
        self.recording_queue = queue.Queue()
        
        self.should_stop_speaking = threading.Event()
        self.is_speaking = threading.Event()
        self.is_listening = threading.Event()
        self.is_processing = threading.Event()
        
        self.conversation_history: List[Dict[str, str]] = []
        self.lock = threading.Lock()


state = State()


# --- Audio Engine (TTS) ---
class AudioEngine:
    def __init__(self):
        self.stream = None
        self.voice = None
        
    def initialize(self):
        print("Loading Piper TTS...")
        self.voice = PiperVoice.load(Config.TTS_MODEL_PATH, use_cuda=False)
        
        self.stream = sd.OutputStream(
            samplerate=Config.SAMPLE_RATE,
            channels=1,
            blocksize=Config.BLOCK_SIZE,
            callback=self._audio_callback,
            dtype=np.float32
        )
        self.stream.start()
        print("✓ Audio engine ready!")
    
    def _audio_callback(self, outdata, frames, time_info, status):
        try:
            chunk = state.audio_queue.get_nowait()
            outdata[:len(chunk)] = chunk
            if len(chunk) < frames:
                outdata[len(chunk):] = 0
        except queue.Empty:
            outdata[:] = 0
    
    def clear_audio_buffer(self):
        while not state.audio_queue.empty():
            try:
                state.audio_queue.get_nowait()
            except queue.Empty:
                break
    
    def shutdown(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()


# --- Speech Recognition Engine ---
class STTEngine:
    def __init__(self):
        self.whisper_model = None
        self.recording_stream = None
        self.audio_buffer = deque(maxlen=int(Config.VAD_SAMPLE_RATE * 10))  # 10 second buffer
        
    def initialize(self):
        print(f"Loading Whisper ({Config.WHISPER_MODEL})...")
        self.whisper_model = whisper.load_model(Config.WHISPER_MODEL)
        print("✓ Speech recognition ready!")
    
    def start_listening(self):
        """Start recording input"""
        def callback(indata, frames, time_info, status):
            if state.is_listening.is_set():
                state.recording_queue.put(indata.copy())
        
        self.recording_stream = sd.InputStream(
            samplerate=Config.VAD_SAMPLE_RATE,
            channels=1,
            callback=callback,
            dtype=np.float32
        )
        self.recording_stream.start()
    
    def detect_speech(self) -> bool:
        """Simple VAD: check if audio exceeds threshold"""
        if len(self.audio_buffer) < Config.VAD_SAMPLE_RATE * Config.MIN_SPEECH_DURATION:
            return False
        
        recent_audio = np.array(list(self.audio_buffer))
        energy = np.abs(recent_audio).mean()
        return energy > Config.VAD_THRESHOLD
    
    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe audio to text"""
        if len(audio_data) < Config.VAD_SAMPLE_RATE * Config.MIN_SPEECH_DURATION:
            return ""
        
        try:
            # Normalize audio
            audio_data = audio_data.flatten()
            
            # Transcribe
            result = self.whisper_model.transcribe(
                audio_data,
                language="en",
                fp16=False,
                temperature=0.0
            )
            
            return result['text'].strip()
        except Exception as e:
            print(f"\n[STT Error: {e}]")
            return ""
    
    def listen_for_speech(self) -> str:
        """Listen for user speech and return transcription"""
        self.audio_buffer.clear()
        state.is_listening.set()
        
        print("\n🎤 Listening... (speak now, pause when done)", end="", flush=True)
        
        silence_start = None
        has_speech = False
        
        while state.is_listening.is_set():
            try:
                audio_chunk = state.recording_queue.get(timeout=0.1)
                self.audio_buffer.extend(audio_chunk.flatten())
                
                # Check for speech
                if self.detect_speech():
                    if not has_speech:
                        has_speech = True
                        print(" [speech detected]", end="", flush=True)
                    silence_start = None
                else:
                    # Track silence after speech
                    if has_speech:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > Config.SILENCE_DURATION:
                            # End of speech detected
                            break
            
            except queue.Empty:
                continue
        
        state.is_listening.clear()
        print("\n⏸️  Processing...", end="", flush=True)
        
        # Transcribe collected audio
        audio_array = np.array(list(self.audio_buffer))
        transcription = self.transcribe(audio_array)
        
        return transcription
    
    def shutdown(self):
        state.is_listening.clear()
        if self.recording_stream:
            self.recording_stream.stop()
            self.recording_stream.close()


# --- LLM Engine ---
class LLMEngine:
    def __init__(self):
        self.llm = None
    
    def initialize(self):
        if not os.path.exists(Config.LLM_MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {Config.LLM_MODEL_PATH}\n"
                "Please download a GGUF model. See README.md"
            )
        
        print("Loading LLM...")
        self.llm = Llama(
            model_path=Config.LLM_MODEL_PATH,
            n_ctx=Config.CONTEXT_WINDOW,
            n_threads=Config.CPU_THREADS,
            n_gpu_layers=Config.GPU_LAYERS,
            verbose=False
        )
        print("✓ LLM ready!")
    
    def build_prompt(self, user_input: str) -> str:
        prompt_parts = [Config.SYSTEM_PROMPT, ""]
        
        with state.lock:
            for msg in state.conversation_history[-Config.MAX_HISTORY:]:
                prompt_parts.append(f"User: {msg['user']}")
                prompt_parts.append(f"Assistant: {msg['assistant']}")
        
        prompt_parts.append(f"User: {user_input}")
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    def stream_response(self, user_input: str) -> str:
        prompt = self.build_prompt(user_input)
        buffer = ""
        full_response = ""
        
        for chunk in self.llm(
            prompt,
            max_tokens=Config.MAX_TOKENS,
            temperature=Config.TEMPERATURE,
            top_p=Config.TOP_P,
            stream=True,
            stop=["User:", "\n\n"]
        ):
            if state.should_stop_speaking.is_set():
                break
            
            token_text = chunk['choices'][0]['text']
            print(token_text, end="", flush=True)
            
            buffer += token_text
            full_response += token_text
            
            if any(p in token_text for p in ".!?"):
                state.text_queue.put(buffer.strip())
                buffer = ""
        
        if buffer.strip() and not state.should_stop_speaking.is_set():
            state.text_queue.put(buffer.strip())
        
        print()
        return full_response.strip()


# --- TTS Worker ---
class TTSWorker:
    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine
        self.thread = None
    
    def start(self):
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
    
    def _worker(self):
        while True:
            phrase = state.text_queue.get()
            
            if phrase is None:
                break
            
            try:
                state.is_speaking.set()
                
                if state.should_stop_speaking.is_set():
                    state.text_queue.task_done()
                    continue
                
                for chunk_obj in self.audio_engine.voice.synthesize(phrase + " "):
                    if state.should_stop_speaking.is_set():
                        break
                    
                    audio_float32 = chunk_obj.audio_float_array * Config.VOLUME_MULTIPLIER
                    
                    for i in range(0, len(audio_float32), Config.BLOCK_SIZE):
                        if state.should_stop_speaking.is_set():
                            break
                        
                        block = audio_float32[i:i+Config.BLOCK_SIZE].reshape(-1, 1)
                        if len(block) < Config.BLOCK_SIZE:
                            block = np.vstack((
                                block,
                                np.zeros((Config.BLOCK_SIZE - len(block), 1), dtype=np.float32)
                            ))
                        state.audio_queue.put(block)
                
            except Exception as e:
                print(f"\n[TTS Error: {e}]")
            finally:
                state.text_queue.task_done()
                state.is_speaking.clear()


# --- Voice Assistant ---
class VoiceAssistant:
    def __init__(self, voice_mode=True):
        self.voice_mode = voice_mode
        self.audio_engine = AudioEngine()
        self.llm_engine = LLMEngine()
        self.stt_engine = STTEngine() if voice_mode else None
        self.tts_worker = TTSWorker(self.audio_engine)
    
    def initialize(self):
        self.audio_engine.initialize()
        self.llm_engine.initialize()
        
        if self.voice_mode:
            self.stt_engine.initialize()
            self.stt_engine.start_listening()
        
        self.tts_worker.start()
    
    def process_input(self, user_input: str):
        state.should_stop_speaking.clear()
        
        print(f"\n💬 You: {user_input}")
        print("🤖 Assistant: ", end="", flush=True)
        
        response = self.llm_engine.stream_response(user_input)
        
        if not state.should_stop_speaking.is_set() and response:
            with state.lock:
                state.conversation_history.append({
                    'user': user_input,
                    'assistant': response
                })
    
    def run_voice_mode(self):
        print("\n" + "="*60)
        print("  🎙️ Voice Assistant (Hands-Free Mode)")
        print("="*60)
        print("\nPress Ctrl+C to exit\n")
        
        try:
            while True:
                # Wait for assistant to finish speaking
                while state.is_speaking.is_set():
                    time.sleep(0.1)
                
                # Listen for user input
                transcription = self.stt_engine.listen_for_speech()
                
                if transcription:
                    self.process_input(transcription)
                else:
                    print(" [no speech detected]")
        
        except KeyboardInterrupt:
            print("\n\n[Stopped by user]")
        finally:
            self.shutdown()
    
    def run_text_mode(self):
        print("\n" + "="*60)
        print("  ⌨️ Voice Assistant (Text Mode)")
        print("="*60)
        print("\nType /help for commands, /quit to exit\n")
        
        try:
            while True:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                if user_input == '/quit':
                    break
                elif user_input == '/help':
                    print("\nCommands: /quit, /clear, /history")
                    continue
                elif user_input == '/clear':
                    with state.lock:
                        state.conversation_history.clear()
                    print("[History cleared]")
                    continue
                elif user_input == '/history':
                    with state.lock:
                        for i, msg in enumerate(state.conversation_history, 1):
                            print(f"{i}. You: {msg['user'][:50]}...")
                    continue
                
                print("Assistant: ", end="", flush=True)
                self.process_input(user_input)
        
        except KeyboardInterrupt:
            print("\n\n[Interrupted]")
        finally:
            self.shutdown()
    
    def shutdown(self):
        print("\nShutting down...")
        state.text_queue.put(None)
        
        if self.stt_engine:
            self.stt_engine.shutdown()
        
        self.audio_engine.shutdown()
        print("Goodbye!")


# --- Entry Point ---
if __name__ == "__main__":
    import sys
    
    voice_mode = "--voice" in sys.argv or "-v" in sys.argv
    
    if voice_mode:
        print("Starting in VOICE MODE (hands-free)")
    else:
        print("Starting in TEXT MODE (use --voice for hands-free)")
    
    assistant = VoiceAssistant(voice_mode=voice_mode)
    assistant.initialize()
    
    if voice_mode:
        assistant.run_voice_mode()
    else:
        assistant.run_text_mode()