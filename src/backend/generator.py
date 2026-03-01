"""
LLM Generator Module

Provides multiple LLM backends for text generation:
- LlamaCppGenerator: Local GGUF model inference
- FallbackExtractorGenerator: Fallback when LLM fails
- LocalQwenGenerator: Alternative local model

Supports GPU acceleration, context management, and structured output
for Data Science and AI question answering.
"""

import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer,
)

try:
    from .device_manager import get_device_config
    from .settings import (
        DEFAULT_GENERATOR_MODEL_PATH as SETTINGS_DEFAULT_GENERATOR_MODEL_PATH,
        DEFAULT_GGUF_PATH as SETTINGS_DEFAULT_GGUF_PATH,
    )
except ImportError:
    from device_manager import get_device_config
    from settings import (
        DEFAULT_GENERATOR_MODEL_PATH as SETTINGS_DEFAULT_GENERATOR_MODEL_PATH,
        DEFAULT_GGUF_PATH as SETTINGS_DEFAULT_GGUF_PATH,
    )


def _setup_llama_cpp_dll_paths() -> None:
    if os.name != "nt":
        return

    add_dir = getattr(os, "add_dll_directory", None)
    if add_dir is None:
        return

    candidates: List[str] = []

    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.append(os.path.join(cuda_path, "bin"))
        candidates.append(os.path.join(cuda_path, "bin", "x64"))

    cudatoolkit_root = os.environ.get("CUDAToolkit_ROOT")
    if cudatoolkit_root:
        candidates.append(os.path.join(cudatoolkit_root, "bin"))
        candidates.append(os.path.join(cudatoolkit_root, "bin", "x64"))

    for v in ("v13.1", "v13.0", "v12.5", "v12.4", "v12.3", "v12.2", "v12.1", "v12.0"):
        candidates.append(
            os.path.join(
                "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA", v, "bin"
            )
        )
        candidates.append(
            os.path.join(
                "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA", v, "bin", "x64"
            )
        )

    for d in candidates:
        try:
            if os.path.isdir(d):
                add_dir(d)
        except Exception:
            pass


_setup_llama_cpp_dll_paths()

try:
    from llama_cpp import Llama
except Exception:
    Llama = None


@dataclass
class SourceSnippet:
    """Source snippet with provenance information"""

    doc_id: str

    chunk_id: str

    text: str

    char_start: int

    char_end: int

    score: float

    source: str


class SimpleProgressStreamer(TextStreamer):

    def __init__(self, tokenizer, skip_prompt=True):

        super().__init__(tokenizer, skip_prompt=skip_prompt)

        self.generated = 0

    def on_finalized_text(self, text: str, stream_end: bool = False):

        self.generated += 1

        if self.generated % 10 == 0 or stream_end:

            print(f"[generator] Tokens generated: {self.generated}")

    def put(self, value):

        pass


DEFAULT_GENERATOR_MODEL_PATH = str(SETTINGS_DEFAULT_GENERATOR_MODEL_PATH)


_CITATION_RE = re.compile(r"``([^`]+?)``")


@dataclass
class RetrievedChunk:

    chunk_id: str

    source: str

    text: str

    score: Optional[float] = None


class LocalQwenGenerator:

    def __init__(
        self,
        model_path: str = DEFAULT_GENERATOR_MODEL_PATH,
        adapter_path: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.5,
        top_p: float = 0.9,
        device: Optional[str] = None,
    ) -> None:

        self.model_path = model_path

        self.max_new_tokens = max_new_tokens

        self.temperature = temperature

        self.top_p = top_p

        device_cfg = get_device_config(device)

        if device_cfg.gpu_required and device_cfg.device != "cuda":

            raise RuntimeError(
                "GPU (CUDA) is required for the generator. Set RAG_DEVICE=cuda and ensure CUDA is available."
            )

        self.device = device_cfg.device

        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        compute_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        quant_config = None

        if self.device == "cuda":

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, use_fast=True, fix_mistral_regex=True
        )

        max_memory = {0: "5.5GiB", "cpu": "10GiB"}

        offload_folder = "offload"

        Path(offload_folder).mkdir(parents=True, exist_ok=True)

        print(f"[startup] Loading merged model from: {model_path}")

        load_kwargs: Dict[str, Any] = {
            "pretrained_model_name_or_path": model_path,
            "local_files_only": True,
            "trust_remote_code": True,
            "attn_implementation": "sdpa" if self.device == "cuda" else "eager",
        }

        allow_cpu_offload = os.environ.get("RAG_ALLOW_CPU_OFFLOAD", "0") in {
            "1",
            "true",
            "True",
        }

        if quant_config is not None:

            load_kwargs["quantization_config"] = quant_config

            if allow_cpu_offload:

                load_kwargs["device_map"] = "auto"

                load_kwargs["max_memory"] = max_memory

                load_kwargs["offload_folder"] = offload_folder

            else:

                load_kwargs["device_map"] = {"": 0}

        else:

            load_kwargs["device_map"] = {"": "cpu"}

        self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

        self.model.eval()

        if self.device == "cuda":
            hf_map = getattr(self.model, "hf_device_map", None)
            if isinstance(hf_map, dict):
                cpu_modules = [k for k, v in hf_map.items() if str(v) == "cpu"]
                print(
                    f"[startup] hf_device_map entries: {len(hf_map)}; cpu modules: {len(cpu_modules)}"
                )
            else:
                print("[startup] hf_device_map not available")

    def _extract_source_snippets(
        self,
        contexts: List[RetrievedChunk],
        answer: str,
        max_snippets: int = 3,
    ) -> List[SourceSnippet]:
        """Extract up to max_snippets most relevant source snippets with character ranges"""

        snippets = []

        # Sort by score (descending) to get most relevant chunks

        sorted_contexts = sorted(
            [c for c in contexts if c.score is not None],
            key=lambda x: x.score,
            reverse=True,
        )

        for ctx in sorted_contexts[:max_snippets]:

            text = ctx.text

            sentences = [s.strip() for s in text.split(".") if s.strip()]

            if sentences:

                best_sentence = max(sentences, key=len)

                char_start = text.find(best_sentence)

                char_end = char_start + len(best_sentence)

                snippets.append(
                    SourceSnippet(
                        doc_id=ctx.chunk_id.split("_chunk_")[0],
                        chunk_id=ctx.chunk_id,
                        text=best_sentence,
                        char_start=char_start,
                        char_end=char_end,
                        score=ctx.score or 0.0,
                        source=ctx.source,
                    )
                )

        return snippets

    def _calculate_confidence(
        self, answer: str, snippets: List[SourceSnippet]
    ) -> float:
        """Calculate confidence score based on answer length and snippet relevance"""

        if not snippets:

            return 0.0

        avg_score = sum(s.score for s in snippets) / len(snippets)

        answer_words = len(answer.split())

        length_factor = min(1.0, answer_words / 100)

        confidence = avg_score * 0.7 + length_factor * 0.3

        return min(1.0, max(0.0, confidence))

    def _extract_citations(self, answer_text: str) -> List[Dict[str, str]]:

        citations: List[Dict[str, str]] = []

        seen: set[tuple[str, str]] = set()

        for m in _CITATION_RE.finditer(answer_text):

            raw = m.group(1).strip()

            if not raw:

                continue

            if ":" not in raw:

                continue

            source, chunk_id = raw.split(":", 1)

            source = source.strip()

            chunk_id = chunk_id.strip()

            if not source or not chunk_id:

                continue

            key = (source, chunk_id)

            if key in seen:

                continue

            seen.add(key)

            citations.append({"source": source, "chunk_id": chunk_id})

        return citations

    def _build_prompt(
        self, question: str, contexts: List[RetrievedChunk]
    ) -> List[Dict[str, str]]:

        ctx_lines: List[str] = []

        for c in contexts:

            header = f"CHUNK chunk_id={c.chunk_id} source={c.source}"

            ctx_lines.append(header)

            ctx_lines.append(c.text)

            ctx_lines.append("")

        ctx_block = "\n".join(ctx_lines).strip()

        system = (
            "You are an expert Data Science and AI assistant. Provide comprehensive, well-structured answers.\n\n"
            "REQUIRED OUTPUT STRUCTURE:\n"
            "1. SUMMARY: Brief 2-3 sentence overview\n"
            "2. TECHNICAL DETAILS: In-depth explanation with key concepts, formulas, or algorithms\n"
            "3. PRACTICAL EXAMPLES: Real-world applications or code snippets if relevant\n"
            "4. SOURCES: Cite context using ``source:chunk_id`` format\n\n"
            "Use the provided context as your primary source. Be thorough and educational."
        )

        user = f"CONTEXT\n{ctx_block}\n\nQUESTION\n{question}\n\nANSWER"

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @torch.no_grad()
    def answer(self, question: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:

        retrieved: List[RetrievedChunk] = []

        for c in contexts:

            retrieved.append(
                RetrievedChunk(
                    chunk_id=str(
                        c.get("chunk_id") if hasattr(c, "get") else c.chunk_id
                    ),
                    source=str(c.get("source") if hasattr(c, "get") else c.source),
                    text=str(c.get("text") if hasattr(c, "get") else c.text),
                    score=c.get("score") if hasattr(c, "get") else c.score,
                )
            )

        messages = self._build_prompt(question=question, contexts=retrieved[:5])

        if hasattr(self.tokenizer, "apply_chat_template"):

            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        else:

            prompt = messages[-1]["content"]

        inputs = self.tokenizer(prompt, return_tensors="pt")

        if self.device == "cuda":

            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        input_len = int(inputs["input_ids"].shape[-1])

        print(
            f"[generator] Starting generation (max_new_tokens={self.max_new_tokens})..."
        )
        start_time = time.time()

        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            pre_gen_mem = torch.cuda.memory_allocated() / 1024**3
            print(f"[generator] GPU memory before generation: {pre_gen_mem:.2f} GB")

        use_streamer = os.environ.get("RAG_STREAM", "0") in {"1", "true", "True"}
        streamer = SimpleProgressStreamer(self.tokenizer) if use_streamer else None

        gen = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )

        elapsed = time.time() - start_time
        if self.device == "cuda":
            post_gen_mem = torch.cuda.memory_allocated() / 1024**3
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"[generator] GPU memory after generation: {post_gen_mem:.2f} GB")
            print(f"[generator] Peak GPU memory during generation: {peak_mem:.2f} GB")

        tok_per_s = (self.max_new_tokens / elapsed) if elapsed > 0 else 0.0
        print(
            f"[generator] Generation complete in {elapsed:.2f}s ({tok_per_s:.1f} tok/s)"
        )

        generated_ids = gen[0][input_len:]

        answer_text = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()

        citations = self._extract_citations(answer_text)

        source_snippets = self._extract_source_snippets(retrieved, answer_text)

        confidence = self._calculate_confidence(answer_text, source_snippets)

        return {
            "answer": answer_text,
            "citations": citations,
            "citations_format": "``source:chunk_id``",
            "sources": [
                {
                    "doc_id": s.doc_id,
                    "chunk_id": s.chunk_id,
                    "text": s.text,
                    "char_start": s.char_start,
                    "char_end": s.char_end,
                    "score": s.score,
                    "source": s.source,
                }
                for s in source_snippets
            ],
            "confidence": float(confidence),
        }


class LlamaCppGenerator:

    def __init__(
        self,
        gguf_path: str,
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.5,
        top_p: float = 0.9,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        device: Optional[str] = None,
        allow_ctx_fallback: bool = True,
        fallback_n_ctx: int = 4096,
    ) -> None:
        device_cfg = get_device_config(device)
        if device_cfg.gpu_required and device_cfg.device != "cuda":
            raise RuntimeError(
                "GPU (CUDA) is required for the llama.cpp generator. Set RAG_DEVICE=cuda and ensure CUDA is available."
            )

        if Llama is None:
            raise RuntimeError("llama-cpp-python is not installed or failed to import")

        self.gguf_path = gguf_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self._lock = threading.Lock()

        self.n_batch = int(os.environ.get("RAG_LLAMA_N_BATCH", "256"))
        self.n_batch = max(32, self.n_batch)
        self.n_batch = min(self.n_batch, n_ctx)
        self.n_batch = min(
            self.n_batch, int(os.environ.get("RAG_LLAMA_N_BATCH_MAX", "512"))
        )
        self.n_threads = int(os.environ.get("RAG_LLAMA_N_THREADS", "0"))
        self.n_threads_batch = int(os.environ.get("RAG_LLAMA_N_THREADS_BATCH", "0"))

        self.prompt_max_chunks = int(os.environ.get("RAG_LLAMA_PROMPT_MAX_CHUNKS", "3"))
        self.prompt_max_chars_per_chunk = int(
            os.environ.get("RAG_LLAMA_PROMPT_MAX_CHARS_PER_CHUNK", "1200")
        )
        self.prompt_max_total_chars = int(
            os.environ.get("RAG_LLAMA_PROMPT_MAX_TOTAL_CHARS", "12000")
        )

        self.verbose = os.environ.get("RAG_LLAMA_VERBOSE", "0") in {"1", "true", "True"}

        def _load(n_ctx_to_use: int) -> Any:
            kwargs: Dict[str, Any] = {
                "model_path": gguf_path,
                "n_ctx": n_ctx_to_use,
                "n_gpu_layers": n_gpu_layers,
                "n_batch": self.n_batch,
                "logits_all": False,
                "embedding": False,
                "verbose": self.verbose,
            }

            if self.n_threads > 0:
                kwargs["n_threads"] = self.n_threads
            if self.n_threads_batch > 0:
                kwargs["n_threads_batch"] = self.n_threads_batch

            return Llama(
                **kwargs,
            )

        try:
            self.llm = _load(n_ctx)
            self.n_ctx = n_ctx
        except Exception as e:
            if not allow_ctx_fallback:
                raise
            print(f"[llama.cpp] Warning: failed to init with n_ctx={n_ctx}: {e}")
            print(f"[llama.cpp] Falling back to n_ctx={fallback_n_ctx}")
            self.llm = _load(fallback_n_ctx)
            self.n_ctx = fallback_n_ctx

        print(
            f"[llama.cpp] Loaded model: n_ctx={self.n_ctx} n_gpu_layers={n_gpu_layers} n_batch={self.n_batch} "
            f"threads={self.n_threads if self.n_threads > 0 else 'default'}"
        )

    def _build_prompt(self, question: str, contexts: List[RetrievedChunk]) -> str:
        ctx_lines: List[str] = []
        total_chars = 0
        for c in contexts:
            header = f"CHUNK chunk_id={c.chunk_id} source={c.source}"
            ctx_lines.append(header)

            txt = c.text
            if len(txt) > self.prompt_max_chars_per_chunk:
                txt = txt[: self.prompt_max_chars_per_chunk]

            remaining = self.prompt_max_total_chars - total_chars
            if remaining <= 0:
                break
            if len(txt) > remaining:
                txt = txt[:remaining]

            ctx_lines.append(txt)
            total_chars += len(txt)
            ctx_lines.append("")

        ctx_block = "\n".join(ctx_lines).strip()

        system = (
            "You are an expert Data Science and AI assistant. Provide comprehensive, well-structured answers.\n\n"
            "REQUIRED OUTPUT STRUCTURE:\n"
            "1. SUMMARY: Brief 2-3 sentence overview\n"
            "2. TECHNICAL DETAILS: In-depth explanation with key concepts, formulas, or algorithms\n"
            "3. PRACTICAL EXAMPLES: Real-world applications or code snippets if relevant\n"
            "4. SOURCES: Cite context using ``source:chunk_id`` format\n\n"
            "Use the provided context as your primary source. Be thorough and educational."
        )

        user = f"CONTEXT\n{ctx_block}\n\nQUESTION\n{question}\n\nANSWER\n"
        return system + "\n\n" + user

    def _extract_citations(self, answer_text: str) -> List[Dict[str, str]]:
        citations: List[Dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for m in _CITATION_RE.finditer(answer_text):
            raw = m.group(1).strip()
            if not raw:
                continue
            if ":" not in raw:
                continue
            source, chunk_id = raw.split(":", 1)
            source = source.strip()
            chunk_id = chunk_id.strip()
            if not source or not chunk_id:
                continue
            key = (source, chunk_id)
            if key in seen:
                continue
            seen.add(key)
            citations.append({"source": source, "chunk_id": chunk_id})
        return citations

    def _extract_source_snippets(
        self,
        contexts: List[RetrievedChunk],
        answer: str,
        max_snippets: int = 3,
    ) -> List[SourceSnippet]:
        snippets: List[SourceSnippet] = []
        sorted_contexts = sorted(
            [c for c in contexts if c.score is not None],
            key=lambda x: x.score,
            reverse=True,
        )

        for ctx in sorted_contexts[:max_snippets]:
            text = ctx.text
            sentences = [s.strip() for s in text.split(".") if s.strip()]
            if sentences:
                best_sentence = max(sentences, key=len)
                char_start = text.find(best_sentence)
                char_end = char_start + len(best_sentence)
                snippets.append(
                    SourceSnippet(
                        doc_id=ctx.chunk_id.split("_chunk_")[0],
                        chunk_id=ctx.chunk_id,
                        text=best_sentence,
                        char_start=char_start,
                        char_end=char_end,
                        score=ctx.score or 0.0,
                        source=ctx.source,
                    )
                )
        return snippets

    def _calculate_confidence(
        self, answer: str, snippets: List[SourceSnippet]
    ) -> float:
        if not snippets:
            return 0.0
        avg_score = sum(s.score for s in snippets) / len(snippets)
        answer_words = len(answer.split())
        length_factor = min(1.0, answer_words / 100)
        confidence = avg_score * 0.7 + length_factor * 0.3
        return min(1.0, max(0.0, confidence))

    def answer(self, question: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        retrieved: List[RetrievedChunk] = []
        for c in contexts:
            retrieved.append(
                RetrievedChunk(
                    chunk_id=str(
                        c.get("chunk_id") if hasattr(c, "get") else c.chunk_id
                    ),
                    source=str(c.get("source") if hasattr(c, "get") else c.source),
                    text=str(c.get("text") if hasattr(c, "get") else c.text),
                    score=c.get("score") if hasattr(c, "get") else c.score,
                )
            )

        prompt = self._build_prompt(
            question=question, contexts=retrieved[: self.prompt_max_chunks]
        )

        print(
            f"[llama.cpp] Starting generation (max_new_tokens={self.max_new_tokens}, n_ctx={self.n_ctx})..."
        )
        start_time = time.time()

        with self._lock:
            try:
                out = self.llm(
                    prompt,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stop=["</s>"],  # Remove source stops to allow longer answers
                )
            except RuntimeError as e:
                msg = str(e)
                if "llama_decode returned" in msg and self.n_batch > 128:
                    retry_batch = 128
                    print(
                        f"[llama.cpp] Warning: decode failed; retrying with safer n_batch={retry_batch}: {e}"
                    )
                    try:
                        if hasattr(self.llm, "n_batch"):
                            self.llm.n_batch = retry_batch
                    except Exception:
                        pass
                    out = self.llm(
                        prompt,
                        max_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        stop=["</s>"],  # Remove source stops to allow longer answers
                    )
                else:
                    raise

        elapsed = time.time() - start_time

        choices = out.get("choices") if isinstance(out, dict) else None
        answer_text = ""
        if isinstance(choices, list) and choices:
            answer_text = str(choices[0].get("text", ""))

        # Fallback: if no choices, try to extract text directly from output
        if not answer_text and isinstance(out, dict):
            # Try common output formats
            if "text" in out:
                answer_text = str(out["text"])
            elif "content" in out:
                answer_text = str(out["content"])
            elif isinstance(out, str):
                answer_text = out
            else:
                # Last resort: convert entire output to string
                answer_text = str(out)
                print(f"[DEBUG] Using fallback extraction: {answer_text[:200]}...")

        answer_text = answer_text.strip()

        usage = out.get("usage") if isinstance(out, dict) else None
        completion_tokens = None
        if isinstance(usage, dict):
            completion_tokens = usage.get("completion_tokens")

        tok_per_s = None
        if isinstance(completion_tokens, int) and completion_tokens > 0:
            tok_per_s = completion_tokens / max(1e-6, elapsed)

        if tok_per_s is not None:
            print(
                f"[llama.cpp] Generation complete in {elapsed:.2f}s ({tok_per_s:.1f} tok/s)"
            )
        else:
            print(f"[llama.cpp] Generation complete in {elapsed:.2f}s")

        citations = self._extract_citations(answer_text)
        source_snippets = self._extract_source_snippets(retrieved, answer_text)
        confidence = self._calculate_confidence(answer_text, source_snippets)

        return {
            "answer": answer_text,
            "citations": citations,
            "citations_format": "``source:chunk_id``",
            "timing": {
                "gen_s": float(elapsed),
                "completion_tokens": (
                    int(completion_tokens)
                    if isinstance(completion_tokens, int)
                    else None
                ),
                "tok_per_s": float(tok_per_s) if tok_per_s is not None else None,
            },
            "sources": [
                {
                    "doc_id": s.doc_id,
                    "chunk_id": s.chunk_id,
                    "text": s.text,
                    "char_start": s.char_start,
                    "char_end": s.char_end,
                    "score": s.score,
                    "source": s.source,
                }
                for s in source_snippets
            ],
            "confidence": float(confidence),
        }


class FallbackExtractorGenerator:
    def __init__(self, max_chars: int = 1600) -> None:
        self.max_chars = max_chars
        self.use_actual_llm = False
        self.actual_generator: Optional[LlamaCppGenerator] = None

        try:
            gguf_env = os.environ.get("RAG_GGUF_PATH", "").strip()
            gguf_path = Path(gguf_env) if gguf_env else SETTINGS_DEFAULT_GGUF_PATH
            if not gguf_path.is_absolute():
                gguf_path = (Path(__file__).resolve().parents[2] / gguf_path).resolve()

            if gguf_path.exists():
                print(
                    "[FallbackExtractorGenerator] Loading LlamaCppGenerator fallback..."
                )
                device_cfg = get_device_config()
                self.actual_generator = LlamaCppGenerator(
                    gguf_path=str(gguf_path),
                    device=device_cfg.device,
                    max_new_tokens=256,
                    n_gpu_layers=28,
                    n_ctx=2048,
                )
                self.use_actual_llm = True
                print(
                    "[FallbackExtractorGenerator] LlamaCppGenerator fallback is ready."
                )
        except Exception as e:
            print(
                "[FallbackExtractorGenerator] Could not initialize "
                f"LlamaCppGenerator fallback: {e}"
            )

    def answer(self, question: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.use_actual_llm and self.actual_generator is not None:
            print("[FallbackExtractorGenerator] Using LlamaCppGenerator fallback.")
            return self.actual_generator.answer(question, contexts)

        merged = "\n\n".join(str(c.get("text", "")) for c in (contexts or []))
        merged = merged.strip()
        snippet = merged[: self.max_chars]

        out = (
            "I couldn't load the full local LLM on this machine. "
            "Below is the most relevant extracted context I found for your question.\n\n"
            f"QUESTION\n{question}\n\nEXTRACTED CONTEXT\n{snippet}"
        )

        return {
            "answer": out,
            "citations": [],
            "sources": [],
            "confidence": 0.2 if snippet else 0.0,
        }


def load_generator(
    model_path: str = DEFAULT_GENERATOR_MODEL_PATH,
) -> LocalQwenGenerator:

    return LocalQwenGenerator(model_path=model_path)
