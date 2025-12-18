# インストールガイド

## 概要

スーパーコンピュータ運用支援RAGシステムのインストール手順書です。本システムはオンプレミス環境での運用を前提としており、Dockerを使用した簡単なデプロイメントが可能です。

## システム要件

### ハードウェア要件

- **CPU**: 4コア以上推奨（最小2コア）
- **メモリ**: 8GB以上推奨（最小4GB）
- **ストレージ**: 50GB以上の空き容量
- **ネットワーク**: インターネット接続（初期セットアップ時のみ）

### ソフトウェア要件

- **OS**: Linux (Ubuntu 20.04+, CentOS 8+) または macOS
- **Docker**: 20.10以降
- **Docker Compose**: 2.0以降
- **Python**: 3.11以降（Dockerを使用しない場合）

## インストール方法

### 方法1: Docker Compose（推奨）

#### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd supercomputer-support-rag
```

#### 2. 環境設定

```bash
# セットアップスクリプトの実行
./scripts/setup.sh --docker

# 環境変数の設定（必要に応じて編集）
cp .env.example .env
nano .env
```

#### 3. システムの起動

```bash
# 開発環境での起動
docker-compose up -d

# 本番環境での起動（Nginxリバースプロキシ付き）
docker-compose -f docker-compose.prod.yml --profile with-nginx up -d
```

#### 4. 動作確認

```bash
# ヘルスチェック
curl http://localhost:8000/health

# システム監視
./scripts/monitor.sh status
```

### 方法2: Python仮想環境

#### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd supercomputer-support-rag
```

#### 2. セットアップ

```bash
# セットアップスクリプトの実行
./scripts/setup.sh

# 仮想環境の有効化
source .venv/bin/activate
```

#### 3. システムの起動

```bash
# APIサーバーの起動
python run_api.py

# または管理インターフェース
python demo_management.py
```

## 設定

### 基本設定（config.yaml）

```yaml
# 主要な設定項目
embedding:
  model_name: "intfloat/multilingual-e5-base"
  model_path: "./models/embedding"

generation:
  model_name: "rinna/japanese-gpt-neox-3.6b"
  model_path: "./models/generation"

crawling:
  base_url: "https://www.cc.kyushu-u.ac.jp/scp/"
  max_depth: 3
  delay: 1.0

search:
  top_k: 5
  min_score: 0.5
```

### 環境変数（.env）

```bash
# API設定
API_PORT=8000
LOG_LEVEL=INFO

# モデル設定
EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-base
GENERATION_MODEL_NAME=rinna/japanese-gpt-neox-3.6b

# パフォーマンス設定
MAX_WORKERS=4
BATCH_SIZE=32
```

## 初期データセットアップ

### 1. 文書の収集

```bash
# 管理インターフェースを使用
python demo_management.py

# または直接APIを使用
curl -X POST http://localhost:8000/api/management/crawl \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.cc.kyushu-u.ac.jp/scp/"}'
```

### 2. インデックスの作成

文書収集後、システムが自動的にインデックスを作成します。進行状況はログで確認できます：

```bash
# ログの確認
tail -f logs/rag_system.log

# または監視スクリプト
./scripts/monitor.sh logs
```

## トラブルシューティング

### よくある問題

#### 1. メモリ不足エラー

**症状**: モデル読み込み時にOOMエラー
**解決策**: 
- Docker Composeのメモリ制限を増加
- より軽量なモデルを使用
- バッチサイズを削減

```yaml
# docker-compose.yml
deploy:
  resources:
    limits:
      memory: 8G
```

#### 2. ポート競合

**症状**: `Address already in use`エラー
**解決策**:
```bash
# ポート使用状況の確認
netstat -tlnp | grep 8000

# 別のポートを使用
export API_PORT=8001
```

#### 3. モデルダウンロードエラー

**症状**: Hugging Faceからのモデルダウンロード失敗
**解決策**:
```bash
# 手動でモデルをダウンロード
python -c "
from transformers import AutoModel, AutoTokenizer
model_name = 'intfloat/multilingual-e5-base'
AutoModel.from_pretrained(model_name, cache_dir='./models/embedding')
AutoTokenizer.from_pretrained(model_name, cache_dir='./models/embedding')
"
```

### ログの確認

```bash
# システムログ
./scripts/monitor.sh logs

# Dockerログ
docker-compose logs -f

# 特定のサービスのログ
docker-compose logs -f rag-system
```

## セキュリティ設定

### 1. ファイアウォール設定

```bash
# UFWを使用する場合
sudo ufw allow 8000/tcp
sudo ufw enable
```

### 2. SSL/TLS設定（本番環境）

```bash
# SSL証明書の配置
mkdir -p nginx/ssl
cp your-cert.pem nginx/ssl/cert.pem
cp your-key.pem nginx/ssl/key.pem

# HTTPS有効化
# nginx/nginx.confのHTTPS設定をコメントアウト
```

### 3. アクセス制限

```nginx
# nginx.confでIP制限
location /api/ {
    allow 192.168.1.0/24;
    deny all;
    # ...
}
```

## バックアップとリストア

### バックアップ

```bash
# 自動バックアップスクリプト
./scripts/backup.sh

# 手動バックアップ
tar -czf backup_$(date +%Y%m%d).tar.gz data/ logs/ config.yaml
```

### リストア

```bash
# バックアップからリストア
tar -xzf backup_20231201.tar.gz
docker-compose restart
```

## アップデート

### システムアップデート

```bash
# 最新コードの取得
git pull origin main

# イメージの再ビルド
docker-compose build

# サービスの再起動
docker-compose up -d
```

### モデルアップデート

```bash
# 新しいモデルのダウンロード
python -c "
from transformers import AutoModel
AutoModel.from_pretrained('new-model-name', cache_dir='./models/embedding')
"

# 設定ファイルの更新
nano config.yaml

# サービスの再起動
docker-compose restart
```

## サポー���

### ログ収集

問題が発生した場合、以下の情報を収集してください：

```bash
# システム情報
./scripts/monitor.sh status > system_info.txt

# ログファイル
cp logs/rag_system.log ./
docker-compose logs > docker_logs.txt

# 設定ファイル
cp config.yaml .env ./
```

### 連絡先

- 技術サポート: [サポート連絡先]
- ドキュメント: [ドキュメントURL]
- 問題報告: [Issue Tracker URL]