# SageMaker 배포 경로 (W8 참고자료)

이 디렉터리는 **W8에서 실제로 검증한 SageMaker 학습→배포→삭제 파이프라인**을
그대로 옮겨온 참고 구현입니다. 이 프로젝트의 K8s 배포(`deploy/k8s/`)와 다른 점:

- 여기 `sagemaker_entry_train.py`는 `sklearn.datasets.make_classification`으로
  만든 **더미 이진분류 데이터**를 학습합니다. 타이타닉 데이터/모델과는 **별개**입니다.
  (SageMaker의 학습 job → 엔드포인트 배포 → 과금 삭제 흐름 자체를 검증하는 것이 목적이었음)
- 이 저장소의 "실제 서빙 경로"는 K8s(`deploy/k8s/deploy.yml`)이고,
  SageMaker는 "클라우드 매니지드 배포로 갈 경우의 대안 경로"를 보여주기 위해 남겨둠.

## 왜 타이타닉 모델로 다시 만들지 않았는가
SageMaker 엔드포인트는 시간당 과금(ml.m5.large 기준 실습 중 약 $0.06~0.1/회)이 발생한다.
동일한 검증(학습→배포→예측→삭제)을 이미 W8에서 완료했으므로, 포트폴리오 통합 단계에서
같은 과금을 또 발생시키기보다 **파이프라인 패턴 자체를 재사용 가능한 형태로 문서화**하는
쪽을 택함 — 필요 시 `entry_point`만 `serve/`의 타이타닉 학습 코드로 바꾸면 그대로 이 저장소에도
적용 가능하다.

## 실행 (참고용, 별도 AWS 비용 발생)
```bash
export SAGEMAKER_ROLE_ARN="arn:aws:iam::<계정ID>:role/<role-name>"   # 본인 계정 값으로 교체
export AWS_REGION="ap-northeast-2"
python deploy_sagemaker.py
```

> 계정 ID/Role ARN은 코드에 하드코딩하지 않는다 — 저장소가 공개되므로
> 반드시 환경변수로 주입한다 (`deploy_sagemaker.py` 참고).
