"""
LLM access and small text-parsing helpers.

API keys are read from environment variables (never hard-coded):

  * ``OPENROUTER_API_KEY`` -- for ``provider="openrouter"`` (default).
  * ``OPENAI_API_KEY``     -- for ``provider="openai"``.

Only the OpenAI Python SDK (``pip install openai``) is required; OpenRouter is
served through the same client with a custom ``base_url``.
"""
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple, Union

from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Clients are created lazily so that importing this module never requires a key.
_clients: Dict[str, OpenAI] = {}


def _get_client(provider: str) -> OpenAI:
    provider = provider.lower()
    if provider in _clients:
        return _clients[provider]

    if provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Export it before running, e.g.\n"
                "  export OPENROUTER_API_KEY=sk-or-..."
            )
        client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)
    elif provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it before running, e.g.\n"
                "  export OPENAI_API_KEY=sk-..."
            )
        client = OpenAI(api_key=key)
    else:
        raise ValueError(f"Unknown provider '{provider}'. Use 'openrouter' or 'openai'.")

    _clients[provider] = client
    return client


def _build_messages(prompt: Union[str, List[str]], system_prompt: str = "") -> List[dict]:
    """Convert a prompt (str, or alternating user/assistant list) into chat messages."""
    if isinstance(prompt, list):
        messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": str(p)}
            for i, p in enumerate(prompt)
        ]
    else:
        messages = [{"role": "user", "content": prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    return messages


def get_llm_response(
    prompt: Union[str, List[str]],
    system_prompt: str = "",
    model: str = "openai/gpt-5.4",
    max_tokens: int = 2048,
    temperature: float = 0.7,
    stop_strs: Optional[List[str]] = None,
    provider: str = "openrouter",
    verbose: bool = False,
    max_retries: int = 3,
    retry_delay: int = 20,
) -> str:
    """Return a chat completion as a stripped string, retrying on transient errors."""
    if verbose:
        head = prompt if isinstance(prompt, str) else prompt[-1]
        print(f"\n--- LLM request ({provider} / {model}) ---")
        print(f"User: {head[:160]}{'...' if len(str(head)) > 160 else ''}")

    messages = _build_messages(prompt, system_prompt)
    client = _get_client(provider)

    base_kwargs = {"model": model, "messages": messages, "stop": stop_strs}
    # Newer "reasoning" models reject max_tokens / non-default temperature; we
    # discover and remember the right call shape on the fly.
    use_completion_tokens = False
    send_temperature = True

    last_err = None
    for attempt in range(max_retries):
        try:
            kwargs = dict(base_kwargs)
            kwargs["max_completion_tokens" if use_completion_tokens else "max_tokens"] = max_tokens
            if send_temperature:
                kwargs["temperature"] = temperature
            resp = client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()
        except Exception as err:  # noqa: BLE001 - surface a clean message after retries
            msg = str(err)
            # Adapt to model-specific parameter constraints without burning a retry.
            if "max_completion_tokens" in msg and not use_completion_tokens:
                use_completion_tokens = True
                continue
            if "temperature" in msg and "unsupported" in msg.lower() and send_temperature:
                send_temperature = False
                continue
            last_err = err
            print(f"[get_llm_response] attempt {attempt + 1}/{max_retries} failed: {err}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    return f"I encountered an error when trying to generate a response: {last_err}"


def get_openai_logit(*args, **kwargs):
    """Logit-based guessing is not used by the released VoI pipeline."""
    raise NotImplementedError(
        "get_openai_logit is unused in this release; call force_guess(get_logits=False)."
    )


# --------------------------------------------------------------------------- #
# Text parsing helpers
# --------------------------------------------------------------------------- #
def parse_guess(raw_guess: str, clean_for_json: bool = True) -> Tuple[str, float, str]:
    """Extract (guess, confidence, cleaned_response) from a model's final answer."""
    cleaned_response = raw_guess
    try:
        if clean_for_json:
            cleaned_response = raw_guess.replace("{{", "{").replace("}}", "}")
        json_match = re.search(r"\{.*\}", cleaned_response, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            return parsed.get("guess", "unknown"), parsed.get("confidence", 0), cleaned_response
        guess_match = re.search(r'guess["\s:]+([^",\n]+)', cleaned_response, re.IGNORECASE)
        if guess_match:
            return guess_match.group(1).strip(), 50, cleaned_response
        raise ValueError("No valid JSON or direct guess found in the response.")
    except Exception as err:  # noqa: BLE001
        print(f"[parse_guess] {err}")
        return "unknown", 0, cleaned_response


def load_json_data(filepath: str):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as err:  # noqa: BLE001
        print(f"[load_json_data] error reading {filepath}: {err}")
        return None


def save_json_data(data, filepath: str, indent: int = 2) -> bool:
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
        return True
    except Exception as err:  # noqa: BLE001
        print(f"[save_json_data] error writing {filepath}: {err}")
        return False


def is_multiple_choice_question(question: str) -> bool:
    return bool(re.search(r"(?:^|\n)[A-F][).] ", question))


def is_yes_no_question(question: str) -> bool:
    patterns = [r"^does .+\?", r"^is .+\?", r"^are .+\?", r"^can .+\?", r"^has .+\?",
                r"^have .+\?", r"^will .+\?", r"^would .+\?", r"^should .+\?", r"^do .+\?"]
    clean = question.lower().strip()
    return any(re.search(p, clean) for p in patterns)


def normalize_text(text: str) -> str:
    return re.sub(r"[.,;:!?\"'()]", "", text).lower().strip()


def check_guess_correctness(guess: str, subject: str) -> bool:
    """A guess matches if it equals, contains, or is contained in the subject."""
    g, s = normalize_text(guess), normalize_text(subject)
    return g == s or s in g or g in s


def extract_guess_from_response(response: str) -> Optional[str]:
    """Extract a guess from a 'My guess is: ...' style response, else None."""
    lower = response.lower()
    if "my guess is:" in lower or "my guess is " in lower:
        parts = lower.split("my guess is:")
        if len(parts) == 1:
            parts = lower.split("my guess is")
        if len(parts) > 1:
            guess = parts[1].strip().strip('."\'?!').lower()
            for word in ("a ", "an ", "the ", "that ", "this "):
                if guess.startswith(word):
                    guess = guess[len(word):]
            return guess.strip()
    return None
