"""LLM and heuristic agents for eval rollouts."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

import torch
from transformers import GenerationConfig

from config import OLLAMA_BASE_URL


class HeuristicAgent:
    """Fast baseline without GPU inference."""

    def decide(self, obs: dict[str, Any]) -> tuple[str, Any]:
        legal = obs.get("legal_actions") or []
        game = obs.get("game", "")
        if not legal:
            return "heuristic fallback", None
        if game in ("pd-classic", "pd-tight", "pd-high-temptation", "ipd-stage"):
            action = "C" if "C" in legal else legal[0]
        elif game == "stag-hunt":
            action = "Stag" if "Stag" in legal else legal[0]
        elif game == "bos":
            action = "Opera" if "Opera" in legal else legal[0]
        elif game == "matching-pennies":
            action = legal[0]
        elif game == "negotiation":
            action = (1, 1, 1)
        elif game == "tic-tac-toe":
            action = legal[0]
        elif game == "auction":
            action = obs.get("private_value", 100)
        elif game == "divide-dollar":
            action = 0.5
        elif game == "p-beauty":
            action = 33
        else:
            action = legal[0]
        return f"heuristic: choose {action}", action


class LoRALLMAgent:
    """Qwen + LoRA agent; output format matches DPO training data."""

    def __init__(self, model, tokenizer, max_new_tokens: int = 192) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def decide(self, obs: dict[str, Any]) -> tuple[str, Any]:
        prompt = obs["prompt"]
        text = _format_llm_prompt(self.tokenizer, prompt)

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        gen_cfg = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        self.model.eval()
        # Kwarg do_sample=False overrides model.generation_config (Llama base defaults to sample).
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                generation_config=gen_cfg,
                do_sample=False,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        reasoning, action = _parse_response(response, obs)
        if action is None:
            action = _fallback_action(obs)
            reasoning = reasoning or f"fallback: {action}"
        return reasoning, action


class OllamaAgent:
    """Frontier eval via local Ollama HTTP API (no HF / bitsandbytes)."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = OLLAMA_BASE_URL,
        max_new_tokens: int = 192,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_new_tokens = max_new_tokens

    def decide(self, obs: dict[str, Any]) -> tuple[str, Any]:
        prompt = obs["prompt"]
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "num_predict": self.max_new_tokens,
                "temperature": 0,
            },
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama request failed ({self.base_url}, model={self.model}). "
                f"Is the server running? "
                f"Try: /scratch/workspace/eunbiyoon_umass_edu-paper/ollama-local/scripts/start.sh"
            ) from exc

        response = body.get("message", {}).get("content", "") or ""
        reasoning, action = _parse_response(response, obs)
        if action is None:
            action = _fallback_action(obs)
            reasoning = reasoning or f"fallback: {action}"
        return reasoning, action


def _format_llm_prompt(tokenizer, prompt: str) -> str:
    """Use chat template when present (Instruct); else raw prompt (pretrained base)."""
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt + "\n"


def _parse_response(text: str, obs: dict[str, Any]) -> tuple[str, Any]:
    thinking = ""
    m_think = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if m_think:
        thinking = m_think.group(1).strip()

    action_raw = None
    m_act = re.search(r"<action>(.*?)</action>", text, re.DOTALL)
    if m_act:
        action_raw = m_act.group(1).strip()
    else:
        for line in reversed(text.strip().splitlines()):
            line = line.strip()
            if line and not line.startswith("<"):
                action_raw = line
                break

    action = _coerce_action(action_raw, obs) if action_raw else None
    return thinking, action


def _coerce_action(raw: str, obs: dict[str, Any]) -> Any:
    legal = obs.get("legal_actions") or []
    game = obs.get("game", "")
    raw = raw.strip()

    if game == "negotiation":
        nums = [int(x) for x in re.findall(r"-?\d+", raw)][:3]
        if len(nums) == 3:
            return tuple(nums)
        return (1, 1, 1)

    if game == "tic-tac-toe":
        nums = re.findall(r"\d+", raw)
        if nums:
            val = int(nums[0])
            if val in legal:
                return val
        return legal[0] if legal else 0

    if game == "auction":
        nums = re.findall(r"\d+", raw)
        return int(nums[0]) if nums else obs.get("private_value", 100)

    if game == "divide-dollar":
        nums = re.findall(r"\d*\.?\d+", raw)
        return float(nums[0]) if nums else 0.5

    if game == "p-beauty":
        nums = re.findall(r"\d+", raw)
        return int(nums[0]) if nums else 33

    for candidate in legal:
        if str(candidate).lower() == raw.lower():
            return candidate
    for candidate in legal:
        if str(candidate).lower() in raw.lower():
            return candidate
    return legal[0] if legal else raw


def _fallback_action(obs: dict[str, Any]) -> Any:
    legal = obs.get("legal_actions") or []
    return legal[0] if legal else None
