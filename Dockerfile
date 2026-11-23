FROM node:22-bookworm-slim AS frontend
WORKDIR /build
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/index.html web/tsconfig.json web/vite.config.ts ./
COPY web/src ./src
COPY web/public ./public
RUN npm run build

FROM python:3.12-slim-bookworm
WORKDIR /app
ENV PYTHONPATH=/app/scripts PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-workflow.txt requirements-web.txt requirements-online.txt ./
RUN pip install --no-cache-dir -r requirements-workflow.txt -r requirements-web.txt -r requirements-online.txt \
    && useradd --create-home --uid 10001 studio
COPY scripts ./scripts
COPY migrations ./migrations
COPY --from=frontend /build/dist ./web/dist
USER studio
CMD ["sh", "-c", "exec uvicorn online_web:app --host 0.0.0.0 --port ${PORT:-8000}"]
