import os
import torch
import json
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.backends.cudnn.enabled = False

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Global cache
_tokenizer = None
_model = None
_pipe = None

def load_llm(model_id: str = MODEL_ID):
    """Load the LLM on CPU and return a text generation pipeline."""
    global _tokenizer, _model, _pipe
    if _pipe is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_id)
        _model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        _pipe = pipeline(
            "text-generation",
            model=_model,
            tokenizer=_tokenizer,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.6,
        )
    return _pipe


# -------------------------------
# Function Tools
# -------------------------------
from modules.search_arxiv import search_arxiv
from modules.calculate import calculate


TOOLS = {
    "search_arxiv": {
        "func": search_arxiv,
        "desc": "Search the Week5 chunks.json for relevant text."
    },
    "calculate": {
        "func": calculate,
        "desc": "Evaluate a math expression safely."
    }
}


SYSTEM_PROMPT = """You are a voice assistant capable of selecting and calling tools.
If the user asks something requiring calculations or database lookup,
you MUST call the correct tool and return the JSON result.
Otherwise, answer normally.
"""


def detect_tool(user_message: str) -> Dict[str, Any]:
    """
    Simple rule-based intent detection.
    In real agents we'd let the model output the function call.
    """
    msg = user_message.lower()

    if "search" in msg or "paper" in msg or "arxiv" in msg:
        return {"tool": "search_arxiv"}

    if "calculate" in msg or "compute" in msg or any(c in msg for c in "+-*/"):
        return {"tool": "calculate"}

    return {"tool": None}


def run_tool(tool_name: str, user_message: str):
    """Execute the selected tool."""
    if tool_name == "search_arxiv":
        query = user_message.replace("search", "").strip()
        return search_arxiv(query)

    if tool_name == "calculate":
        expr = user_message.replace("calculate", "").strip()
        return calculate(expr)

    return None


def chat_reply(history: List[Dict[str, str]]) -> str:
    """
    Main chat function supporting function calling.
    """
    user_message = history[-1]["content"]

    # Step 1: detect whether to call a tool
    intent = detect_tool(user_message)

    if intent["tool"] is not None:
        result = run_tool(intent["tool"], user_message)
        return f"[TOOL RESULT]\n{json.dumps(result, indent=2)}"

    # Step 2: normal LLM reply
    llm = load_llm()
    prompt = SYSTEM_PROMPT + "\n\n"

    for msg in history:
        prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"

    prompt += "Assistant:"

    output = llm(prompt)[0]["generated_text"]
    return output.split("Assistant:")[-1].strip()
