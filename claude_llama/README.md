# Streaming TTS with llama.cpp Setup Guide

## Installation

### 1. Install Python Dependencies

```bash
pip install llama-cpp-python sounddevice numpy piper-tts
```

**For GPU acceleration (optional but recommended):**
```bash
# CUDA 12.x
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# CUDA 11.x
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11/bin/nvcc" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Metal (macOS)
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### 2. Download Models

#### Piper TTS Model (Voice)
```bash
# Download Amy voice model
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json
```

#### LLM Model (choose one)

**Option 1: Llama 3.2 3B Instruct (Recommended - Best quality)**
```bash
mkdir -p models
cd models
# Download Q4_K_M quantization (4.3GB)
wget https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf -O llama-3.2-3b-instruct-q4_k_m.gguf
```

**Option 2: Phi-3.5 Mini Instruct (Smaller, faster - 3.8B params)**
```bash
mkdir -p models
cd models
wget https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf
```

**Option 3: Qwen2.5 3B Instruct (Good balance)**
```bash
mkdir -p models
cd models
wget https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf
```

**Option 4: TinyLlama 1.1B (Fastest, lowest quality)**
```bash
mkdir -p models
cd models
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

### 3. Update Model Path in Script

Edit `streaming_tts_llamacpp.py` and set the correct path:
```python
LLM_MODEL_PATH = "./models/llama-3.2-3b-instruct-q4_k_m.gguf"  # or your chosen model
```

## Usage

```bash
python streaming_tts_llamacpp.py
```

Type your questions and press ENTER. The LLM will respond with streaming text and speech.

## Performance Tuning

### CPU Optimization
```python
llm = Llama(
    model_path=LLM_MODEL_PATH,
    n_ctx=2048,
    n_threads=8,        # Increase for more CPU cores
    n_batch=512,        # Batch size for prompt processing
    n_gpu_layers=0,
)
```

### GPU Optimization (NVIDIA)
```python
llm = Llama(
    model_path=LLM_MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=35,    # Offload all layers to GPU (adjust based on VRAM)
    n_batch=512,
)
```

### GPU Optimization (macOS Metal)
```python
llm = Llama(
    model_path=LLM_MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=1,     # Metal uses different layer system
)
```

## Quantization Types Explained

- **Q4_K_M**: 4-bit quantization, medium quality (recommended)
- **Q5_K_M**: 5-bit, better quality, slightly larger
- **Q8_0**: 8-bit, highest quality, 2x size
- **Q4_0**: 4-bit, smallest, lower quality
- **Q2_K**: 2-bit, very small but quality suffers

## Memory Requirements

| Model | Q4_K_M | Q5_K_M | Q8_0 |
|-------|--------|--------|------|
| 1B params | ~700MB | ~900MB | ~1.4GB |
| 3B params | ~2GB | ~2.5GB | ~4GB |
| 7B params | ~4GB | ~5GB | ~8GB |

## Troubleshooting

### "Model not found" error
- Check the path in `LLM_MODEL_PATH`
- Ensure you downloaded the model to the correct location

### Slow performance
- Increase `n_threads` (but not beyond your CPU core count)
- Use GPU acceleration if available
- Try a smaller model or Q4 quantization

### Audio issues
- Check `sounddevice.query_devices()` for correct audio device
- Adjust `blocksize` if you hear crackling (try 2048 or 4096)

### Out of memory
- Use a smaller model (1B or 3B instead of 7B)
- Use more aggressive quantization (Q4_0 or Q2_K)
- Reduce `n_ctx` (context window)

## Key Improvements Over PyTorch Version

1. **3-5x faster inference** on CPU
2. **50-70% less memory usage** with quantization
3. **No 5GB+ PyTorch/CUDA install** required
4. **Better model selection** (Llama 3.2, Phi-3.5, Qwen)
5. **Easy GPU offloading** with `n_gpu_layers`