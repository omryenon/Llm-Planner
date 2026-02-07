import os, json
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional

from app.prompts import SYSTEM
from app.ollama_manager import build_manager_from_env

load_dotenv()

app = FastAPI(title="LLM Planner")
ollama = build_manager_from_env()

class PlanRequest(BaseModel):
    car: str
    start: dict
    end: dict
    candidates: list
    conflicts_summary: Optional[dict] = None


@app.on_event("startup")
def _startup():
    ollama.start_if_needed()
    ollama.ensure_model(ollama.model)

@app.on_event("shutdown")
def _shutdown():
  ollama.shutdown()


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
""".strip()

    raw = ollama.generate(prompt=prompt, system=SYSTEM, timeout=120)

    try:
        return json.loads(raw)
    except Exception:
        return {"error": "LLM did not return valid JSON", "raw": raw}
