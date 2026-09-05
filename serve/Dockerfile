# W10 통합 — 서빙 이미지.
# 빌드 (레포 루트에서): docker build -t titanic-api:latest -f serve/Dockerfile serve/
# 실행: docker run --rm -p 8000:8000 titanic-api:latest

FROM python:3.14-slim
WORKDIR /app

# requirements 먼저 COPY → pip install. 코드 한 줄 고칠 때마다
# pip install을 처음부터 다시 하지 않도록 레이어 캐시를 살린다.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# --host 를 빼거나 127.0.0.1 로 두면 -p 8000:8000 을 줘도 호스트에서 접속이 안 된다.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
