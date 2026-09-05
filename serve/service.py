"""W10 통합 — BentoML 서비스화 (W6 대안 서빙 경로).

준비 : python save_model.py   (model.joblib -> BentoML 모델 스토어에 등록)
실행 : bentoml serve service:TitanicService
"""

import bentoml
import pandas as pd
from pydantic import BaseModel

FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare"]


class Passenger(BaseModel):
    pclass: int
    sex: int
    age: float
    sibsp: int
    parch: int
    fare: float


@bentoml.service
class TitanicService:
    model_ref = bentoml.models.get("titanic_clf:latest")

    def __init__(self):
        self.model = bentoml.sklearn.load_model(self.model_ref)

    @bentoml.api
    def predict(self, p: Passenger) -> dict:
        X = pd.DataFrame([p.model_dump()])[FEATURES]
        return {"survived": int(self.model.predict(X)[0])}
