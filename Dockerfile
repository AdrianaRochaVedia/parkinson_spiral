FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]
