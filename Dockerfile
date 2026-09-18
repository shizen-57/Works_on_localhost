# syntax=docker/dockerfile:1
#
# Multi-stage build. Runtime image: Python 3.12-slim, non-root, binds
# 0.0.0.0:$PORT, no secrets baked in (LLM_API_KEY etc. are read from the
# environment at container start only -- see .env.example). CMD uses shell
# form deliberately: exec-form ["uvicorn", ..., "--port", "$PORT"] does NOT
# expand environment variables (plan_review.md flagged this).

FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.12-slim
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser data ./data

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=8000

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT','8000') + '/health', timeout=3)"

CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
