FROM registry.falahtech.com/base/lang-heavy:py311 AS builder

WORKDIR /app

COPY app/requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

FROM python:3.11-slim
COPY --from=builder /install /usr/local
COPY app /app
WORKDIR /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]