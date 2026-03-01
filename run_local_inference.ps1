Param(
  [switch]$BuildIndex,
  [string]$AdapterPath = ""
)

$ErrorActionPreference = "Stop"

$projRoot = Split-Path -Parent $PSScriptRoot
$modelsDir = if (Test-Path "$projRoot\models") { "$projRoot\models" } else { "$projRoot\shared\models\models" }
$dataDir = if (Test-Path "$projRoot\data") { "$projRoot\data" } else { "$projRoot\shared\data\data" }

if ($BuildIndex) {
  Write-Host "[run_local_inference] Building FAISS index..."
  python -m src.backend.ingest build
}

$adapterArg = ""
if ($AdapterPath -ne "") {
  $adapterArg = "--adapter-path `"$AdapterPath`""
} elseif (Test-Path "$projRoot\models\qwen-ds-adapter") {
  $adapterArg = "--adapter-path `"$projRoot\models\qwen-ds-adapter`""
} elseif (Test-Path "$projRoot\eval\qwen-ds-adapter") {
  $adapterArg = "--adapter-path `"$projRoot\eval\qwen-ds-adapter`""
} elseif (Test-Path "$projRoot\models\final_adapter") {
  $adapterArg = "--adapter-path `"$projRoot\models\final_adapter`""
}

Write-Host "[run_local_inference] Starting API server..."
Write-Host "Adapter arg: $adapterArg"

if ($adapterArg -ne "") {
  $env:RAG_ADAPTER_PATH = $AdapterPath
  if ($AdapterPath -eq "") {
    if (Test-Path "$projRoot\models\qwen-ds-adapter") {
      $env:RAG_ADAPTER_PATH = "$projRoot\models\qwen-ds-adapter"
    } elseif (Test-Path "$projRoot\eval\qwen-ds-adapter") {
      $env:RAG_ADAPTER_PATH = "$projRoot\eval\qwen-ds-adapter"
    } elseif (Test-Path "$projRoot\models\final_adapter") {
      $env:RAG_ADAPTER_PATH = "$projRoot\models\final_adapter"
    }
  }
}

# Llama.cpp environment variables for Qwen2.5-1.5B-Instruct GGUF
$env:RAG_LLM_BACKEND = "llama_cpp"
$env:RAG_GGUF_PATH = "$modelsDir\qwen2_5_1_5b_instruct\gguf\qwen2.5-1.5b-instruct-q4_k_m.gguf"
$env:RAG_EMBEDDING_MODEL_PATH = "$modelsDir\bge-m3"
$env:RAG_INDEX_DIR = "$dataDir\indexes"
$env:RAG_MAX_NEW_TOKENS = "256"
$env:RAG_LLAMA_N_CTX = "2048"
$env:RAG_LLAMA_FALLBACK_N_CTX = "1024"
$env:RAG_LLAMA_N_GPU_LAYERS = "28"
$env:RAG_LLAMA_N_BATCH = "256"
$env:RAG_LLAMA_N_BATCH_MAX = "256"
$env:RAG_LLAMA_VERBOSE = "0"
$env:RAG_FAST = "1"
$env:RAG_USE_RERANKER = "0"
$env:RAG_RETRIEVAL_K = "50"
$env:RAG_HYBRID_ALPHA = "0.6"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
$env:CUDAToolkit_ROOT = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64;$env:PATH"

python -m uvicorn src.backend.server:app --host 127.0.0.1 --port 8000
