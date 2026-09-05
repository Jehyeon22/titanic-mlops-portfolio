"""Evidently 데이터 드리프트 리포트 생성.

실행 (레포 루트에서, make_drift_data.py 이후): python monitoring/make_drift_report.py
결과: monitoring/drift_report.html
"""

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

reference = pd.read_csv("monitoring/reference.csv")
current = pd.read_csv("monitoring/recent.csv")

report = Report([DataDriftPreset()])
snapshot = report.run(current_data=current, reference_data=reference)
snapshot.save_html("monitoring/drift_report.html")

print("saved -> monitoring/drift_report.html")
