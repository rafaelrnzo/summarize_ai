FROM registry.falahtech.com/base/lang-heavy:py311 AS builder

WORKDIR /app

COPY app/requirements.txt .

RUN pip install --upgrade pip && \
    pip install --target=/install -r requirements.txt

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/usr/local/lib/python3.11/site-packages"

WORKDIR /app

COPY --from=builder /install /usr/local/lib/python3.11/site-packages/

COPY --from=builder /install/bin/ /usr/local/bin/

COPY app /app

EXPOSE 8003
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]