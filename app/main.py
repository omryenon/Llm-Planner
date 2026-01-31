import os, json, requests
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional


load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

app = FastAPI(title="LLM Planner")

class PlanRequest(BaseModel):
    # minimal inputs for now
    car: str
    start: dict
    end: dict
    candidates: list  # list of {algorithm, metrics, ...}
    conflicts_summary: Optional[dict] = None

def call_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["response"]

@app.post("/plan")
def plan(req: PlanRequest):
    prompt = f"""
{{
  "task": "Choose best route candidate and propose one NEW variant (custom weights). Return JSON only.",
  "inputs": {{
    "car": "{req.car}",
    "start": {json.dumps(req.start)},
    "end": {json.dumps(req.end)},
    "candidates": {json.dumps(req.candidates)},
    "conflicts_summary": {json.dumps(req.conflicts_summary or {})}
  }},
  "output_schema": {{
    "best_candidate_algorithm": "string",
    "reason": "string",
    "proposed_custom_algorithm": {{
      "name": "string",
      "base_algorithm": "astar",
      "weights": {{
        "w_road": "number",
        "w_offroad": "number",
        "w_slope": "number",
        "w_danger": "number",
        "w_landcover": "number",
        "w_conflict": "number"
      }}
    }}
  }}
}}
"""
    raw = call_ollama(prompt)

    # enforce JSON only
    try:
        data = json.loads(raw)
    except Exception:
        return {"error": "LLM did not return valid JSON", "raw": raw}

    return data