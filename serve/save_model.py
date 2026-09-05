"""model.joblib -> BentoML 모델 스토어 등록 (service.py 실행 전 1회)."""

import bentoml
import joblib

pipe = joblib.load("model.joblib")
bentoml.sklearn.save_model("titanic_clf", pipe)
