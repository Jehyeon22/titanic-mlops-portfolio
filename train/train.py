"""W10 통합 — 타이타닉 모델 학습 (W2/W3: sklearn Pipeline + MLflow 실험관리).

실행 (레포 루트에서): python train/train.py
결과:
  - serve/model.joblib   (W6 서빙이 로드할 아티팩트)
  - ./mlruns/            (MLflow 로컬 트래킹 스토어 — `mlflow ui` 로 확인)
"""

import argparse
import os

import joblib
import mlflow
import mlflow.sklearn
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare"]
TARGET = "survived"


def load_data():
    df = sns.load_dataset("titanic")
    X = df[FEATURES].copy()
    y = df[TARGET]
    X["sex"] = (X["sex"] == "male").astype(int)  # male=1, female=0
    return X, y


def build_pipeline():
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),  # age 결측 177건 -> 중앙값
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="serve/model.joblib",
        help="모델 저장 경로 (기본: 레포 루트 기준 serve/model.joblib)",
    )
    parser.add_argument(
        "--experiment", default="titanic-portfolio", help="MLflow 실험 이름"
    )
    args = parser.parse_args()

    mlflow.set_experiment(args.experiment)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipe = build_pipeline()

    with mlflow.start_run():
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("features", FEATURES)
        mlflow.log_param("impute_strategy", "median")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        # mlflow 3.x 기본 직렬화(skops)가 이 스택(numpy/sklearn 버전)에서
        # numpy.dtype을 신뢰되지 않은 타입으로 막는다 → cloudpickle로 저장.
        mlflow.sklearn.log_model(pipe, "model", serialization_format="cloudpickle")

        run_id = mlflow.active_run().info.run_id

    print("accuracy:", acc)
    print("f1:", f1)
    print("학습 시 컬럼 순서:", list(X.columns))
    print("MLflow run_id:", run_id)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    joblib.dump(pipe, args.output)
    print("saved ->", args.output)


if __name__ == "__main__":
    main()
