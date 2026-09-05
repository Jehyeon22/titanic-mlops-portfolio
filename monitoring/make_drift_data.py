"""기준(reference) / 최근(current) 데이터 생성 — Evidently 드리프트 리포트용.

실행 (레포 루트에서): python monitoring/make_drift_data.py
"""

import seaborn as sns

FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare"]

df = sns.load_dataset("titanic")
X = df[FEATURES].copy()
X["sex"] = (X["sex"] == "male").astype(int)  # train.py 와 동일 인코딩

# 학습 시점 데이터 역할
X.to_csv("monitoring/reference.csv", index=False)

# 최근 데이터 역할 - 일부러 나이를 +20 해서 분포를 어긋나게 함 (드리프트 시뮬레이션)
current = X.copy()
current["age"] = current["age"] + 20
current.to_csv("monitoring/recent.csv", index=False)

print("saved monitoring/reference.csv / monitoring/recent.csv")
