# スーパーコンピュータ運用支援RAGシステム - Dockerfile
FROM python:3.11-slim

# 作業ディレクトリの設定
WORKDIR /app

# システムパッケージのインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pythonの依存関係をコピーしてインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY . .

# データディレクトリの作成
RUN mkdir -p data/documents data/vectors data/logs data/processing data/website_structure logs

# モデルディレクトリの作成
RUN mkdir -p models/embedding models/generation

# 非rootユーザーの作成
RUN useradd -m -u 1000 raguser && \
    chown -R raguser:raguser /app

USER raguser

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ポートの公開
EXPOSE 8000

# アプリケーションの起動
CMD ["python", "run_api.py"]
