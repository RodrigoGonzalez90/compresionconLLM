"""
Abstracción del backend LM para obtener distribuciones de probabilidad.

OllamaBackend: llama la API /v1/completions de Ollama (logprobs no siempre disponibles)
LlamaCppBackend: usa llama-cpp-python directamente, con soporte nativo de logprobs
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import httpx

log = logging.getLogger(__name__)

TOP_K = 64

# Mapeo nombre-ollama → (repo HuggingFace, filename GGUF)
GGUF_MODELS: dict[str, tuple[str, str]] = {
    "qwen2.5:1.5b":  ("Qwen/Qwen2.5-1.5B-Instruct-GGUF",       "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
    "phi3:mini":     ("microsoft/Phi-3-mini-4k-instruct-gguf",   "Phi-3-mini-4k-instruct-q4.gguf"),
    "llama3.2:1b":   ("bartowski/Llama-3.2-1B-Instruct-GGUF",    "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
    "tinyllama:1b":  ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",  "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
}


def normalize_token(tok: str) -> str:
    """Convierte marcadores de espacio sentencepiece/GPT-2 a espacio real."""
    return tok.replace("▁", " ").replace("Ġ", " ")


class LLMBackend:
    """Interfaz base: dado un contexto, retorna los TOP_K tokens más probables."""

    def top_tokens(self, context: str) -> List[Tuple[str, float]]:
        """
        Retorna lista de (token_str_raw, logprob) ordenada por probabilidad desc.
        token_str_raw puede tener prefijos sentencepiece (▁, Ġ).
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Resetea el estado interno (KV cache) antes de un nuevo mensaje."""
        pass

    def next_token(self, text: str) -> str:
        """
        Retorna el primer token del texto según el vocabulario propio del modelo.
        Garantiza que los boundaries OOV sean consistentes con el LM.
        Fallback: primer carácter.
        """
        return text[0] if text else ""

    def tokenize_with_ids(self, text: str) -> List[Tuple[str, int]]:
        """
        Tokeniza el texto y retorna pares (token_str_normalizado, vocab_id).
        Retorna lista vacía si el backend no soporta IDs (Ollama).
        """
        return []

    def id_to_token(self, token_id: int) -> str:
        """Convierte un vocab ID al string del token (normalizado). Fallback: ''."""
        return ""

    @property
    def vocab_size(self) -> int:
        """Tamaño del vocabulario. 0 si el backend no lo expone (Ollama)."""
        return 0


class OllamaBackend(LLMBackend):
    """Backend Ollama vía /v1/completions. logprobs no disponibles en todos los modelos."""

    def __init__(self, host: str, modelo: str):
        self.host = host.rstrip("/")
        self.modelo = modelo

    def top_tokens(self, context: str) -> List[Tuple[str, float]]:
        try:
            resp = httpx.post(
                f"{self.host}/v1/completions",
                json={
                    "model":       self.modelo,
                    "prompt":      context,
                    "max_tokens":  1,
                    "temperature": 0,
                    "logprobs":    TOP_K,
                    "stream":      False,
                },
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()
            logprobs_obj = data["choices"][0].get("logprobs")
            if not logprobs_obj:
                log.debug("Ollama logprobs=null para %s", self.modelo)
                return []
            raw = (logprobs_obj.get("top_logprobs") or [{}])[0]
            if isinstance(raw, dict):
                items = list(raw.items())
            elif isinstance(raw, list):
                items = [(t["token"], t["logprob"]) for t in raw]
            else:
                return []
            return sorted(items, key=lambda x: x[1], reverse=True)
        except Exception as exc:
            log.debug("OllamaBackend.top_tokens error: %s", exc)
            return []


class LlamaCppBackend(LLMBackend):
    """
    Backend llama-cpp-python. Soporta logprobs nativamente.
    Descarga automáticamente el GGUF desde HuggingFace Hub si no está en caché.
    """

    def __init__(self, modelo: str, cache_dir: str = "/app/.cache/huggingface"):
        from llama_cpp import Llama

        entry = GGUF_MODELS.get(modelo)
        if not entry:
            raise ValueError(
                f"Modelo '{modelo}' no mapeado. Opciones: {list(GGUF_MODELS)}"
            )
        repo_id, filename = entry
        log.info("LlamaCpp: cargando %s / %s ...", repo_id, filename)
        self._llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            n_ctx=512,
            n_threads=1,   # determinismo: mismo orden de ops en encoder y decoder
            logits_all=True,
            verbose=False,
        )
        log.info("LlamaCpp: modelo listo.")

    def reset(self) -> None:
        """Limpia el KV cache para garantizar determinismo entre mensajes."""
        self._llm.reset()

    def next_token(self, text: str) -> str:
        """Primer token del texto según el tokenizador interno del modelo."""
        try:
            ids = self._llm.tokenize(text.encode("utf-8"), add_bos=False, special=False)
            if ids:
                raw = self._llm.detokenize([ids[0]])
                return raw.decode("utf-8", errors="replace")
        except Exception as exc:
            log.debug("next_token error: %s", exc)
        return text[0] if text else ""

    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokeniza el texto completo con el tokenizador del modelo.
        Retorna la lista de strings de tokens (normalizados, sin prefijos ▁/Ġ).
        """
        return [tok for tok, _ in self.tokenize_with_ids(text)]

    def tokenize_with_ids(self, text: str) -> List[Tuple[str, int]]:
        """Tokeniza y retorna pares (token_str_normalizado, vocab_id)."""
        ids = self._llm.tokenize(text.encode("utf-8"), add_bos=False, special=False)
        result = []
        for id_ in ids:
            raw = self._llm.detokenize([id_])
            result.append((normalize_token(raw.decode("utf-8", errors="replace")), id_))
        return result

    def id_to_token(self, token_id: int) -> str:
        """Convierte un vocab ID al string del token normalizado."""
        raw = self._llm.detokenize([token_id])
        return normalize_token(raw.decode("utf-8", errors="replace"))

    @property
    def vocab_size(self) -> int:
        return self._llm.n_vocab()

    def top_tokens(self, context: str) -> List[Tuple[str, float]]:
        try:
            out = self._llm(
                context,
                max_tokens=1,
                temperature=0,
                logprobs=TOP_K,
            )
            logprobs_obj = out["choices"][0].get("logprobs") or {}
            top = (logprobs_obj.get("top_logprobs") or [{}])[0]
            if isinstance(top, dict):
                # Convertir numpy.float32 → float Python
                items = [(k, float(v)) for k, v in top.items()]
            elif isinstance(top, list):
                # Formato alternativo: [(token, logprob), ...]
                items = [(t, float(p)) for t, p in top]
            else:
                log.warning("top_logprobs formato inesperado: %s", type(top))
                items = []
            return sorted(items, key=lambda x: x[1], reverse=True)
        except Exception as exc:
            log.warning("LlamaCppBackend.top_tokens error: %s", exc)
            return []


def build_backend(backend: str, modelo: str, ollama_host: str) -> LLMBackend:
    """Factory: crea el backend correcto según la variable de entorno BACKEND."""
    if backend == "llamacpp":
        return LlamaCppBackend(modelo=modelo)
    return OllamaBackend(host=ollama_host, modelo=modelo)
