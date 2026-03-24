FROM python:3.11-slim

WORKDIR /app

# Step 1 - Install torch from PyTorch index
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 300 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.4.0+cpu"

# Step 2 - Install everything else from PyPI
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 300 -r requirements.txt

COPY app/ .
COPY data/ ./data/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]