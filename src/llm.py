"""LLM 호출 래퍼.

우선순위는 providers.gen_backend() 가 정한다. **온프렘이 기본이므로 Ollama 가 1순위다.**
Ollama 가 떠 있으면 키가 있어도 외부로 나가지 않는다.

Upstage 와 OpenAI 는 OpenAI 호환이라 같은 경로를 쓴다.
"""
import json
import os
import urllib.request

from config import (CHAT_MODEL_ANTHROPIC, CHAT_MODEL_OLLAMA, CHAT_MODEL_OPENAI,
                    CHAT_MODEL_UPSTAGE, EMBED_API, OLLAMA_HOST, OLLAMA_NUM_GPU)
from providers import gen_backend


class NoLLMKey(RuntimeError):
    pass


def _post(url, payload, headers, timeout=120):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _openai_compatible(base_url, key, model, system, user, max_tokens):
    body = _post(
        f"{base_url}/chat/completions",
        {"model": model, "max_tokens": max_tokens, "temperature": 0,
         "messages": [{"role": "system", "content": system},
                      {"role": "user", "content": user}]},
        {"Authorization": f"Bearer {key}"})
    return body["choices"][0]["message"]["content"]


def complete(system: str, user: str, max_tokens: int = 800) -> str:
    backend = gen_backend()

    if backend == "ollama":
        # 온프렘 경로. 스트리밍을 끄고 한 번에 받는다.
        options = {"temperature": 0, "num_predict": max_tokens}
        if OLLAMA_NUM_GPU is not None:
            # 설정된 경우에만 보낸다. 안 보내면 Ollama 가 알아서 정한다.
            options["num_gpu"] = int(OLLAMA_NUM_GPU)
        body = _post(
            f"{OLLAMA_HOST}/api/chat",
            {"model": CHAT_MODEL_OLLAMA, "stream": False, "options": options,
             "messages": [{"role": "system", "content": system},
                          {"role": "user", "content": user}]},
            {}, timeout=600)   # 로컬 CPU 추론은 느리다. 약 8 토큰/초
        return body["message"]["content"]

    if backend == "anthropic":
        body = _post(
            "https://api.anthropic.com/v1/messages",
            {"model": CHAT_MODEL_ANTHROPIC, "max_tokens": max_tokens,
             "system": system, "messages": [{"role": "user", "content": user}]},
            {"x-api-key": os.environ["ANTHROPIC_API_KEY"],
             "anthropic-version": "2023-06-01"})
        return "".join(b.get("text", "") for b in body["content"])

    if backend == "upstage":
        return _openai_compatible(
            EMBED_API["upstage"]["base_url"], os.environ["UPSTAGE_API_KEY"],
            CHAT_MODEL_UPSTAGE, system, user, max_tokens)

    if backend == "openai":
        return _openai_compatible(
            EMBED_API["openai"]["base_url"], os.environ["OPENAI_API_KEY"],
            CHAT_MODEL_OPENAI, system, user, max_tokens)

    raise NoLLMKey(
        "생성 백엔드가 없습니다. Ollama 를 띄우거나, .env.local 에 "
        "ANTHROPIC_API_KEY / UPSTAGE_API_KEY / OPENAI_API_KEY 중 하나를 넣으십시오.")
