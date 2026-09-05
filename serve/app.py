"""W10 통합 — 타이타닉 추론 API (W4/W6: FastAPI 서빙).

실행 : uvicorn app:app --host 0.0.0.0 --port 8000   (serve/ 디렉터리에서)
확인 : http://localhost:8000/docs
판정 : /health -> {"status":"ok"}
       /predict 3등급·남성·22세·저운임 -> survived 0
                1등급·여성·30세·고운임 -> survived 1
"""

import joblib
import pandas as pd
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

# 학습(train/train.py) 때와 컬럼 순서가 같아야 한다.
# sklearn 파이프라인은 위치 기반이라 순서가 틀리면 에러 없이 조용히 틀린 예측을 낸다.
FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare"]

app = FastAPI(title="titanic-api")

Instrumentator().instrument(app).expose(app)  # W7: Prometheus 스크레이프용 /metrics

# 모듈 레벨 로드. 요청마다 로드하면 응답이 수십~수백배 느려진다.
model = joblib.load("model.joblib")


class Passenger(BaseModel):
    pclass: int = Field(..., ge=1, le=3, description="객실 등급 1/2/3")
    sex: int = Field(..., ge=0, le=1, description="male=1, female=0")
    age: float = Field(..., ge=0, le=120)
    sibsp: int = Field(..., ge=0, description="동승 형제/배우자 수")
    parch: int = Field(..., ge=0, description="동승 부모/자녀 수")
    fare: float = Field(..., ge=0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(p: Passenger):
    X = pd.DataFrame([p.model_dump()])[FEATURES]
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])
    return {"survived": pred, "probability": round(proba, 4)}
