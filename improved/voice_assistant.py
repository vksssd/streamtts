import sounddevice as sd
import numpy as np
import threading
import queue
import time
from piper import PiperVoice
from llama_cpp import Llama
from typing import List, Dict
import os

# --- Configuration ---
class Config:
    # Paths
    TTS_MODEL_PATH = "en_US-amy-medium.onnx"
    LLM_MODEL_PATH = "./models/llama-3.2-3b-instruct-q4_k_m.gguf"
    
    # Audio settings
    SAMPLE_RATE = 22050
    BLOCK_SIZE = 1024
    AUDIO_QUEUE_SIZE = 100
    VOLUME_MULTIPLIER = 1.5
    
    # LLM settings
    CONTEXT_WINDOW = 2048
    MAX_TOKENS = 200
    TEMPERATURE = 0.7
    TOP_P = 0.9
    CPU_THREADS = 4
    GPU_LAYERS = 0  # Set to 35+ for GPU acceleration
    
    # Conversation settings
    MAX_HISTORY = 10  # Keep last 10 exchanges
    SYSTEM_PROMPT = "You are a helpful, friendly assistant. Keep responses concise and conversational."


# --- Global State ---
class State:
    def __init__(self):
        self.audio_queue = queue.Queue(maxsize=Config.AUDIO_QUEUE_SIZE)
        self.text_queue = queue.Queue()
        self.should_stop_speaking = threading.Event()
        self.is_speaking = threading.Event()
        self.conversation_history: List[Dict[str, str]] = []
        self.lock = threading.Lock()


state = State()


# --- Audio Engine ---
class AudioEngine:
    def __init__(self):
        self.stream = None
        self.voice = None
        
    def initialize(self):
        """Initialize TTS and audio stream"""
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
        print("Audio engine ready!")
    
    def _audio_callback(self, outdata, frames, time_info, status):
        """Callback for audio stream"""
        try:
            chunk = state.audio_queue.get_nowait()
            outdata[:len(chunk)] = chunk
            if len(chunk) < frames:
                outdata[len(chunk):] = 0
        except queue.Empty:
            outdata[:] = 0
    
    def clear_audio_buffer(self):
        """Clear pending audio"""
        while not state.audio_queue.empty():
            try:
                state.audio_queue.get_nowait()
            except queue.Empty:
                break
    
    def shutdown(self):
        """Clean shutdown"""
        if self.stream:
            self.stream.stop()
            self.stream.close()


# --- LLM Engine ---
class LLMEngine:
    def __init__(self):
        self.llm = None
    
    def initialize(self):
        """Load LLM model"""
        if not os.path.exists(Config.LLM_MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {Config.LLM_MODEL_PATH}\n"
                "Please download a GGUF model. See README.md for instructions."
            )
        
        print("Loading LLM with llama.cpp...")
        self.llm = Llama(
            model_path=Config.LLM_MODEL_PATH,
            n_ctx=Config.CONTEXT_WINDOW,
            n_threads=Config.CPU_THREADS,
            n_gpu_layers=Config.GPU_LAYERS,
            verbose=False
        )
        print("LLM ready!")
    
    def build_prompt(self, user_input: str) -> str:
        """Build prompt with conversation history"""
        prompt_parts = [Config.SYSTEM_PROMPT, ""]
        
        # Add conversation history
        with state.lock:
            for msg in state.conversation_history[-Config.MAX_HISTORY:]:
                prompt_parts.append(f"User: {msg['user']}")
                prompt_parts.append(f"Assistant: {msg['assistant']}")
        
        # Add current question
        prompt_parts.append(f"User: {user_input}")
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    def stream_response(self, user_input: str) -> str:
        """Generate streaming response"""
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
            # Check if we should stop
            if state.should_stop_speaking.is_set():
                print("\n[Interrupted]")
                break
            
            token_text = chunk['choices'][0]['text']
            print(token_text, end="", flush=True)
            
            buffer += token_text
            full_response += token_text
            
            # Send to TTS on sentence boundary
            if any(p in token_text for p in ".!?"):
                state.text_queue.put(buffer.strip())
                buffer = ""
        
        # Send remaining text
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
        """Start TTS worker thread"""
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
    
    def _worker(self):
        """Background thread for text-to-speech"""
        while True:
            phrase = state.text_queue.get()
            
            if phrase is None:
                break
            
            try:
                state.is_speaking.set()
                
                # Check if we should stop before synthesizing
                if state.should_stop_speaking.is_set():
                    state.text_queue.task_done()
                    continue
                
                # Synthesize speech
                for chunk_obj in self.audio_engine.voice.synthesize(phrase + " "):
                    # Check for interruption
                    if state.should_stop_speaking.is_set():
                        break
                    
                    audio_float32 = chunk_obj.audio_float_array * Config.VOLUME_MULTIPLIER
                    
                    # Split into blocks
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


# --- Conversation Manager ---
class ConversationManager:
    def __init__(self, llm_engine: LLMEngine, audio_engine: AudioEngine):
        self.llm_engine = llm_engine
        self.audio_engine = audio_engine
    
    def process_input(self, user_input: str):
        """Process user input and generate response"""
        # Reset interruption flag
        state.should_stop_speaking.clear()
        
        print("Assistant: ", end="", flush=True)
        response = self.llm_engine.stream_response(user_input)
        
        # Save to history if not interrupted
        if not state.should_stop_speaking.is_set() and response:
            with state.lock:
                state.conversation_history.append({
                    'user': user_input,
                    'assistant': response
                })
    
    def interrupt(self):
        """Interrupt current speech"""
        if state.is_speaking.is_set():
            print("\n[Interrupting...]")
            state.should_stop_speaking.set()
            
            # Clear queues
            while not state.text_queue.empty():
                try:
                    state.text_queue.get_nowait()
                except queue.Empty:
                    break
            
            self.audio_engine.clear_audio_buffer()
            
            # Wait for speech to stop
            time.sleep(0.2)
            state.should_stop_speaking.clear()
    
    def clear_history(self):
        """Clear conversation history"""
        with state.lock:
            state.conversation_history.clear()
        print("\n[Conversation history cleared]")
    
    def show_history(self):
        """Display conversation history"""
        with state.lock:
            if not state.conversation_history:
                print("\n[No conversation history]")
                return
            
            print("\n--- Conversation History ---")
            for i, msg in enumerate(state.conversation_history, 1):
                print(f"\n{i}. You: {msg['user']}")
                print(f"   Assistant: {msg['assistant'][:100]}{'...' if len(msg['assistant']) > 100 else ''}")
            print("\n--- End of History ---\n")


# --- Main Application ---
class VoiceAssistant:
    def __init__(self):
        self.audio_engine = AudioEngine()
        self.llm_engine = LLMEngine()
        self.tts_worker = TTSWorker(self.audio_engine)
        self.conversation_manager = None
    
    def initialize(self):
        """Initialize all components"""
        self.audio_engine.initialize()
        self.llm_engine.initialize()
        self.tts_worker.start()
        self.conversation_manager = ConversationManager(
            self.llm_engine,
            self.audio_engine
        )
    
    def show_help(self):
        """Show available commands"""
        print("\n--- Commands ---")
        print("  /clear    - Clear conversation history")
        print("  /history  - Show conversation history")
        print("  /stop     - Interrupt current speech")
        print("  /help     - Show this help")
        print("  /quit     - Exit the program")
        print("  Ctrl+C    - Interrupt and stop")
        print("----------------\n")
    
    def run(self):
        """Main interaction loop"""
        print("\n" + "="*60)
        print("  Voice Assistant with llama.cpp")
        print("="*60)
        self.show_help()
        
        try:
            while True:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    cmd = user_input.lower()
                    
                    if cmd == '/quit':
                        print("\nGoodbye!")
                        break
                    elif cmd == '/clear':
                        self.conversation_manager.clear_history()
                    elif cmd == '/history':
                        self.conversation_manager.show_history()
                    elif cmd == '/stop':
                        self.conversation_manager.interrupt()
                    elif cmd == '/help':
                        self.show_help()
                    else:
                        print(f"Unknown command: {user_input}")
                    continue
                
                # Process normal input
                self.conversation_manager.process_input(user_input)
        
        except KeyboardInterrupt:
            print("\n\n[Interrupted by user]")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Clean shutdown"""
        print("\nShutting down...")
        state.text_queue.put(None)
        self.audio_engine.shutdown()
        print("Goodbye!")


# --- Entry Point ---
if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.initialize()
    assistant.run()