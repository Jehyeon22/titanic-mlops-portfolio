import os

import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn

REGION = os.environ.get("AWS_REGION", "ap-northeast-2")
# 계정 ID/Role ARN을 코드에 하드코딩하지 않는다 — 환경변수로 주입.
# 실행 전: export SAGEMAKER_ROLE_ARN="arn:aws:iam::<account-id>:role/<role-name>"
ROLE = os.environ["SAGEMAKER_ROLE_ARN"]

boto_session = boto3.Session(region_name=REGION)
sess = sagemaker.Session(boto_session=boto_session)
bucket = sess.default_bucket()

s3_train_path = sess.upload_data(
    path="train.csv",
    bucket=bucket,
    key_prefix="w8-lab/data"
)

print("업로드 완료:", s3_train_path)

sk = SKLearn(
    entry_point="sagemaker_entry_train.py",
    role=ROLE,
    instance_type="ml.m5.large",
    framework_version="1.2-1",
    py_version="py3",
    sagemaker_session=sess
)

sk.fit({"train": s3_train_path})

predictor = sk.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large"
)

with open("endpoint_name.txt", "w") as f:
    f.write(predictor.endpoint_name)
print("엔드포인트 생성완료:", predictor.endpoint_name)

sample = [[0.1, 0.2, 0.3, 0.4]]
print("예측결과", predictor.predict(sample))