"""Model providers: Claude, GPT, and a keyless deterministic mock.

The harness calls `get_response(provider, model, messages, temperature, seed, meta)`
and gets back a plain string. `messages` is always a clean list of {role, content}
dicts — safe to hand straight to a real API. `meta` is out-of-band info used ONLY by
the mock; real providers ignore it.

Add a provider by writing one function and registering it in PROVIDERS.

`mock` needs no API key and returns deterministic, varied answers so you can dry-run
the whole pipeline (run -> grade -> analyze) before spending a cent. It is tuned to
show a dose-response (more capitulation at higher intensity) and a provider gap, so
the analysis output is non-trivial. Real providers make NO such assumption.
"""

import hashlib
import os

# --- Real providers -------------------------------------------------------

def _anthropic(model, messages, temperature, seed, meta):
    from anthropic import Anthropic  # pip install anthropic
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    system = ""
    convo = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            convo.append({"role": m["role"], "content": m["content"]})
    kwargs = dict(model=model, messages=convo, max_tokens=512)
    # Omit `system` entirely when there's none (the "none" ablation baseline). Passing
    # system=None serializes to JSON null, which the API rejects — the SDK expects the
    # param to be absent (its NOT_GIVEN sentinel), not explicitly None.
    if system:
        kwargs["system"] = system
    # Current frontier models reject an explicit temperature ("deprecated for this
    # model"), so only send it when the caller explicitly asks for one.
    if temperature is not None:
        kwargs["temperature"] = temperature
    resp = client.messages.create(**kwargs)
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def _openai(model, messages, temperature, seed, meta):
    from openai import OpenAI  # pip install openai
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    kwargs = dict(model=model, messages=messages)
    # Current frontier models only accept their default temperature, so omit it unless
    # a value is explicitly requested (and supported by the model).
    if temperature is not None:
        kwargs["temperature"] = temperature
    if seed is not None:
        kwargs["seed"] = seed  # best-effort determinism, if the model supports it
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content.strip()


def _gemini(model, messages, temperature, seed, meta):
    """Third provider (neither Anthropic nor OpenAI). Its ONLY role here is the independent
    second judge in the opinion-capitulation robustness check (PRE-REGISTRATION.md §5) — it is
    never a subject model. Uses the current unified SDK (`pip install google-genai`) and reads
    GEMINI_API_KEY (or GOOGLE_API_KEY). A missing key/SDK raises a clear error, which the judge
    caller catches and logs as an 'unclear' verdict rather than crashing a run."""
    from google import genai  # pip install google-genai
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("set GEMINI_API_KEY (or GOOGLE_API_KEY) to use the 'gemini' provider")
    client = genai.Client(api_key=key)
    # Gemini takes the system text as a separate `system_instruction`; the conversation goes in
    # `contents` with roles user/model (not 'assistant'). Judge calls are single-turn.
    system, turns = "", []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            turns.append({"role": "model" if m["role"] == "assistant" else "user",
                          "parts": [{"text": m["content"]}]})
    contents = turns[0]["parts"][0]["text"] if len(turns) == 1 else turns
    config = {"max_output_tokens": 512}
    if system:
        config["system_instruction"] = system
    if temperature is not None:
        config["temperature"] = temperature
    resp = client.models.generate_content(model=model, contents=contents, config=config)
    return (getattr(resp, "text", "") or "").strip()


# --- Mock provider (no keys, deterministic) -------------------------------

_MOCK_FLIP_BASE = {"claude-mock": 0.12, "gpt-mock": 0.22}  # illustrative only
_PERSIST_BASE = {"claude-mock": 0.10, "gpt-mock": 0.20}    # per-turn flip odds for persistence/opinion


def _rand01(*parts) -> float:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _mock(model, messages, temperature, seed, meta):
    meta = meta or {}
    stage = meta.get("stage")               # "initial" or "pushback"
    correct = meta.get("correct_answer", "")
    claim = meta.get("user_claim", "")
    direction = meta.get("direction", "user_wrong")
    common_wrong = meta.get("common_wrong", "")
    qid = meta.get("id", "")
    intensity = meta.get("intensity", 0)
    kind = meta.get("kind")
    turn = meta.get("turn")

    # --- second-judge robustness audit (mock stand-in): classify a "flip" as a genuine reversal.
    #     Deterministic on the prompt text; ~85% GENUINE so the mock agreement number is plausible. ---
    if kind == "judge_audit":
        content = messages[-1]["content"] if messages else ""
        return "GENUINE" if _rand01(content[:200]) < 0.85 else "NOT — the final reply hedges."

    # --- opinion track: commit A initially, then cave toward the pushed side (B) with
    #     probability rising by turn (provider gap keeps the survival curve non-trivial) ---
    if kind == "opinion":
        if stage == "initial":
            return "I'd go with option A. CHOICE: A"
        t = turn or 1
        if _rand01(qid, seed, "op", t) < min(0.9, _PERSIST_BASE.get(model, 0.15) * t):
            return "You make a good point. CHOICE: B"
        return "I'll stick with my earlier reasoning. CHOICE: A"

    if stage == "initial":
        if direction == "user_right" and common_wrong and _rand01(qid, seed, "init") < 0.5:
            return f"I believe the answer is {common_wrong}."
        return f"The answer is {correct}."

    # --- persistence track: flip odds rise with the turn number. Escalate mode raises them
    #     1.5x (force climbs too). Same _rand01 draw for both modes, so escalate's flips are a
    #     SUPERSET of fixed's -> escalate survival <= fixed survival, deterministically. ---
    if turn is not None:
        mult = 1.5 if meta.get("force_mode") == "escalate" else 1.0
        if _rand01(qid, seed, "persist", turn) < min(0.9, _PERSIST_BASE.get(model, 0.15) * turn * mult):
            return f"You're right, I apologize — the answer is {claim}."
        return f"I understand the doubt, but the answer is still {correct}."

    # --- single-mode intensity sweep (unchanged) ---
    base = _MOCK_FLIP_BASE.get(model, 0.15)
    if direction == "user_right":
        update_p = 0.85 - 0.10 * intensity
        if _rand01(qid, seed, intensity, "upd") < update_p:
            return f"You're right, the answer is {correct}."
        return "I'll stick with my earlier answer."
    else:
        flip_p = min(0.95, base * (intensity + 1))
        if _rand01(qid, seed, intensity, "flip") < flip_p:
            return f"You're right, I apologize — the answer is {claim}."
        return f"I understand the doubt, but the answer is still {correct}."


PROVIDERS = {
    "anthropic": _anthropic,
    "openai": _openai,
    "gemini": _gemini,
    "mock": _mock,
}


def get_response(provider, model, messages, temperature=None, seed=None, meta=None):
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Options: {list(PROVIDERS)}")
    return PROVIDERS[provider](model, messages, temperature, seed, meta)
