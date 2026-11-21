"""FastAPI detection endpoint.   Run:  uvicorn api:app --port 8000"""
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import config as C
import detector

app = FastAPI(title="SQL Injection Detection API", version="1.0")


class DetectRequest(BaseModel):
    payload: str
    algorithm: str = "xgboost"            # svm | xgboost | mlp | all
    endpoint: Optional[str] = None
    method: Optional[str] = None
    source: str = "api"
    true_label: Optional[str] = None
    true_category: Optional[str] = None
    timestamp: Optional[str] = None
    request_id: Optional[str] = None
    log: bool = True


class BatchRequest(BaseModel):
    items: List[DetectRequest]


def _run(items, log):
    try:
        return detector.run_detection([i.model_dump() for i in items], log=log)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/health")
def health():
    return {"status": "ok", "models_ready": detector.models_ready(), "algorithms": list(C.ALGOS)}


@app.post("/detect")
def detect(req: DetectRequest):
    """Classify one request payload (raw SQL, JSON body, or a full HTTP request)."""
    return _run([req], req.log)


@app.post("/detect/batch")
def detect_batch(req: BatchRequest):
    return _run(req.items, all(i.log for i in req.items))
