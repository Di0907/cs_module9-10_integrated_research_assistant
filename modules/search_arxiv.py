from typing import List
import json
import os

def search_arxiv(query: str):
    """Search Week5 chunks.json file for matching text."""

    # build absolute path to data_cscl
    base_dir = os.path.dirname(os.path.dirname(__file__))  # .../voice-agent
    chunks_path = os.path.join(base_dir, "data_cscl", "chunks.json")

    results = []
    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        for c in chunks:
            if query.lower() in c["text"].lower():
                results.append(c)

    except Exception as e:
        return {"error": str(e)}

    return {"matches": results[:5]}
