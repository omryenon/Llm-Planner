# llm_planner.py
import os, time, json
import requests
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi import status

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")

# חשוב: אל תתן למודל "לחפור"
NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "80"))
READ_TIMEOUT = float(os.getenv("OLLAMA_READ_TIMEOUT", "300"))   # שניות
CONNECT_TIMEOUT = float(os.getenv("OLLAMA_CONNECT_TIMEOUT", "5"))

app = FastAPI(title="LLM Planner", version="1.0")

class Point(BaseModel):
  lat: float
  lng: float

class Candidate(BaseModel):
  algorithm: str
  metrics: Dict[str, Any] = Field(default_factory=dict)

class PlanRequest(BaseModel):
  car: str
  start: Point
  end: Point
  candidates: List[Candidate] = Field(default_factory=list)
  conflicts_summary: Optional[Dict[str, float]] = None

SYSTEM = (
    "You are a routing assistant. "
    "Return ONLY valid JSON. No explanations. "
    "JSON schema: {\"pick\":\"<algorithm>\",\"reason\":\"<short>\"} "
    "Pick must be one of the provided candidates' algorithm values."
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
  return JSONResponse(
    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    content={"detail": exc.errors(), "body": exc.body},
  )

@app.on_event("startup")
def warmup():
  try:
    payload = {
      "model": OLLAMA_MODEL,
      "prompt": "Return JSON: {\"ok\":true}",
      "stream": False,
      "options": {
          "temperature": 0,
          "top_p": 1,
          "num_predict": 20,
      },
    }
    requests.post(
      f"{OLLAMA_URL}/api/generate",
      json=payload,
      timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
    )
    print("LLM warmup done")
  except Exception as e:
    print("LLM warmup failed:", e)


def _extract_first_json(text: str) -> Dict[str, Any]:
    # robust: model sometimes adds text after JSON
    i = text.find("{")
    if i == -1:
        raise ValueError("No JSON found")
    for j in range(len(text), i, -1):
        chunk = text[i:j]
        try:
            return json.loads(chunk)
        except Exception:
            continue
    raise ValueError("Could not parse JSON")

def call_ollama(system: str, prompt: str) -> Dict[str, Any]:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 1,
            "num_predict": NUM_PREDICT,
        },
    }
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json=payload,
        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
    )
    if not r.ok:
        raise RuntimeError(f"Ollama {r.status_code}: {r.text}")
    return r.json()

@app.get("/health")
def health():
    # בודק שיש Ollama, בלי לטעון מודלים כבדים
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=(1, 2))
        return {"ok": r.ok, "ollama_status": r.status_code}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/plan")
def plan(req: PlanRequest, request: Request):
    t0 = time.time()

    def compute_score(c, conflicts_summary):
      length = float(c.metrics.get("length_m", 1e12))
      conflict = float((conflicts_summary or {}).get(c.algorithm, 0.0))
      return length + conflict

    cands = [
      {
        "algorithm": c.algorithm,
        "length_m": float(c.metrics.get("length_m", 0.0)),
        "conflict": float((req.conflicts_summary or {}).get(c.algorithm, 0.0)),
        "score": compute_score(c, req.conflicts_summary),
      }
      for c in req.candidates
    ]

    prompt_obj = {
        "car": req.car,
        "start": req.start.model_dump(),
        "end": req.end.model_dump(),
        "candidates": cands,
        "instruction": "Pick the best algorithm. Prefer lowest score. Output JSON only.",
    }
    prompt = json.dumps(prompt_obj, ensure_ascii=False)

    try:
        oll = call_ollama(SYSTEM, prompt)
        raw = oll.get("response", "")
        parsed = _extract_first_json(raw)
        pick = parsed.get("pick")

        valid = {c["algorithm"] for c in cands}
        if pick not in valid:
            # fallback: lowest score
            best = min(cands, key=lambda x: x["score"]) if cands else None
            pick = best["algorithm"] if best else None
            parsed = {"pick": pick, "reason": "fallback_lowest_score"}

        return {
            "pick": parsed.get("pick"),
            "reason": parsed.get("reason", ""),
            "_meta": {
                "prompt_len": len(prompt),
                "elapsed_s": round(time.time() - t0, 3),
                "model": OLLAMA_MODEL,
                "num_predict": NUM_PREDICT,
            },
        }

    except Exception as e:
        # לא מפיל את המערכת – מחזיר error נקי
        best = min(cands, key=lambda x: x["score"])["algorithm"] if cands else None
        return {
            "pick": best,
            "reason": "fallback_llm_error",
            "error": str(e),
            "_meta": {
                "prompt_len": len(prompt),
                "elapsed_s": round(time.time() - t0, 3),
                "model": OLLAMA_MODEL,
                "num_predict": NUM_PREDICT,
            },
        }