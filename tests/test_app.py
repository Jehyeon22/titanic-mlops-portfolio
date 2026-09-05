"""서빙 API 스모크 테스트 (CI에서 실행).

serve/app.py가 모듈 레벨에서 "model.joblib"을 상대경로로 로드하므로,
serve/ 를 cwd로 바꾼 뒤 import한다. train/train.py 를 먼저 실행해서
serve/model.joblib이 존재해야 한다.
"""

import os
import sys

from fastapi.testclient import TestClient

SERVE_DIR = os.path.join(os.path.dirname(__file__), "..", "serve")
sys.path.insert(0, SERVE_DIR)
os.chdir(SERVE_DIR)

from app import app  # noqa: E402

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_class3_male_low_fare_dies():
    resp = client.post(
        "/predict",
        json={"pclass": 3, "sex": 1, "age": 22, "sibsp": 1, "parch": 0, "fare": 7.25},
    )
    assert resp.status_code == 200
    assert resp.json()["survived"] == 0


def test_predict_class1_female_high_fare_survives():
    resp = client.post(
        "/predict",
        json={"pclass": 1, "sex": 0, "age": 30, "sibsp": 0, "parch": 0, "fare": 100},
    )
    assert resp.status_code == 200
    assert resp.json()["survived"] == 1
