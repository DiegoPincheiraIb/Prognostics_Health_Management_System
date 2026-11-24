FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml ./
COPY rul ./rul
ARG TORCH_WHEEL_URL
RUN if [ -n "$TORCH_WHEEL_URL" ]; then pip install --no-cache-dir "$TORCH_WHEEL_URL"; \
    else pip install --no-cache-dir 'torch>=2.2,<3' --index-url https://download.pytorch.org/whl/cpu; fi \
    && pip install --no-cache-dir '.[wandb]'
COPY dashboard.py ./
CMD ["uvicorn", "rul.api:app", "--host", "0.0.0.0", "--port", "8000"]