# 運用マニュアル

## 概要

スーパーコンピュータ運用支援RAGシステムの日常運用手順書です。システム管理者向けの運用タスク、監視方法、メンテナンス手順を説明します。

## 日常運用タスク

### システム監視

#### 1. システム状態の確認

```bash
# 総合監視ダッシュボード
./scripts/monitor.sh

# 個別チェック
./scripts/monitor.sh status    # システム状態
./scripts/monitor.sh health    # APIヘルスチェック
./scripts/monitor.sh resources # リソース使用状況
./scripts/monitor.sh data      # データファイル状況
```

#### 2. ログ監視

```bash
# リアルタイムログ監視
tail -f logs/rag_system.log

# エラーログの確認
grep -i "error\|exception\|failed" logs/rag_system.log | tail -20

# 監視スクリプトでのログ確認
./scripts/monitor.sh logs
```

#### 3. パフォーマンス監視

**監視項目**:
- CPU使用率（80%以下を維持）
- メモリ使用率（85%以下を維持）
- ディスク使用率（90%以下を維持）
- API応答時間（5秒以下を維持）

```bash
# リソース使用状況の詳細確認
docker stats scp-rag-system

# API応答時間の測定
time curl -s http://localhost:8000/health
```

### データ管理

#### 1. 文書の更新

```bash
# 管理インターフェースの起動
python demo_management.py

# または直接APIを使用
curl -X POST http://localhost:8000/api/management/crawl \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.cc.kyushu-u.ac.jp/scp/"}'
```

#### 2. インデックスの管理

```bash
# インデックス状態の確認
curl http://localhost:8000/api/management/status

# インデックスの再構築
curl -X POST http://localhost:8000/api/management/reindex
```

#### 3. データベースのメンテナンス

```bash
# ベクトルデータベースの最適化
python -c "
from src.services.vector_database import VectorDatabase
db = VectorDatabase()
db.optimize()
"
```

## 定期メンテナンス

### 日次タスク

#### 1. バックアップ

```bash
# 自動バックアップの実行
./scripts/backup.sh

# バックアップの確認
ls -la backups/
```

#### 2. ログローテーション

```bash
# ログファイルのサイズ確認
du -h logs/

# 古いログの圧縮（7日以上前）
find logs/ -name "*.log" -mtime +7 -exec gzip {} \;
```

#### 3. システムヘルスチェック

```bash
# 総合ヘルスチェック
python system_integration_check.py

# API機能テスト
python -m pytest tests/test_api.py -v
```

### 週次タスク

#### 1. パフォーマンス分析

```bash
# 検索ログの分析
python -c "
import json
from collections import Counter

# 検索クエリの分析
with open('logs/search.log', 'r') as f:
    queries = [json.loads(line)['query'] for line in f if 'query' in line]
    
print('Top 10 queries:')
for query, count in Counter(queries).most_common(10):
    print(f'{count}: {query}')
"
```

#### 2. データ品質チェック

```bash
# 文書数の確認
find data/documents -name "*.json" | wc -l

# ベクトルファイルの整合性確認
python -c "
import numpy as np
import os

vector_dir = 'data/vectors'
for file in os.listdir(vector_dir):
    if file.endswith('.npy'):
        try:
            vectors = np.load(os.path.join(vector_dir, file))
            print(f'{file}: {vectors.shape}')
        except Exception as e:
            print(f'Error loading {file}: {e}')
"
```

#### 3. セキュリティ更新

```bash
# Dockerイメージの更新
docker-compose pull
docker-compose up -d

# Python依存関係の更新確認
pip list --outdated
```

### 月次タスク

#### 1. 容量管理

```bash
# ディスク使用量の詳細分析
du -sh data/* logs/* models/*

# 古いバックアップの削除
find backups/ -name "*.tar.gz" -mtime +30 -delete
```

#### 2. モデル更新の検討

```bash
# 新しいモデルの確認
python -c "
from huggingface_hub import list_models
models = list_models(filter='japanese')
for model in models[:5]:
    print(model.modelId)
"
```

## アラート対応

### 高CPU使用率

**症状**: CPU使用率が80%を超える
**対応手順**:

1. 原因の特定
```bash
# プロセス確認
top -p $(pgrep -f run_api.py)

# 処理中のリクエスト確認
curl http://localhost:8000/api/management/status
```

2. 対処法
```bash
# ワーカー数の調整
export MAX_WORKERS=2
docker-compose restart

# または処理の一時停止
docker-compose pause rag-system
```

### 高メモリ使用率

**症状**: メモリ使用率が85%を超える
**対応手順**:

1. メモリ使用量の確認
```bash
# コンテナのメモリ使用量
docker stats --no-stream scp-rag-system

# プロセス別メモリ使用量
ps aux | grep python | sort -k4 -nr
```

2. 対処法
```bash
# バッチサイズの削減
export BATCH_SIZE=16
docker-compose restart

# 不要なキャッシュのクリア
docker system prune -f
```

### API応答遅延

**症状**: API応答時間が5秒を超える
**対応手順**:

1. 原因の調査
```bash
# 処理時間の詳細確認
curl -w "@curl-format.txt" -s http://localhost:8000/health

# ログでの処理時間確認
grep "processing_time" logs/rag_system.log | tail -10
```

2. 対処法
```bash
# 検索パラメータの調整
# config.yamlでtop_kを削減
search:
  top_k: 3  # 5から3に削減
  min_score: 0.6  # 0.5から0.6に上昇

# サービス再起動
docker-compose restart
```

### ディスク容量不足

**症状**: ディスク使用率が90%を超える
**対応手順**:

1. 容量使用状況の確認
```bash
# ディレクトリ別使用量
du -sh data/* logs/* models/* backups/*

# 大きなファイルの特定
find . -size +100M -type f -exec ls -lh {} \;
```

2. 対処法
```bash
# ログファイルの圧縮
gzip logs/*.log

# 古いバックアップの削除
find backups/ -mtime +7 -delete

# 不要なDockerイメージの削除
docker image prune -f
```

## 障害対応

### サービス停止

**対応手順**:

1. 状態確認
```bash
# コンテナ状態の確認
docker-compose ps

# ログの確認
docker-compose logs --tail=50 rag-system
```

2. 復旧手順
```bash
# サービス再起動
docker-compose restart

# 完全な再起動が必要な場合
docker-compose down
docker-compose up -d
```

### データ破損

**対応手順**:

1. 破損範囲の確認
```bash
# データファイルの整合性チェック
python verify_offline.py

# ベクトルファイルの確認
python -c "
import numpy as np
import os
for file in os.listdir('data/vectors'):
    if file.endswith('.npy'):
        try:
            np.load(f'data/vectors/{file}')
            print(f'{file}: OK')
        except:
            print(f'{file}: CORRUPTED')
"
```

2. 復旧手順
```bash
# バックアップからの復旧
./scripts/backup.sh  # 現在の状態をバックアップ
tar -xzf backups/latest_backup.tar.gz

# インデックスの再構築
curl -X POST http://localhost:8000/api/management/reindex
```

## 設定変更

### モデル変更

```bash
# 1. 新しいモデルのダウンロード
python -c "
from transformers import AutoModel, AutoTokenizer
model_name = 'new-model-name'
AutoModel.from_pretrained(model_name, cache_dir='./models/embedding')
AutoTokenizer.from_pretrained(model_name, cache_dir='./models/embedding')
"

# 2. 設定ファイルの更新
nano config.yaml

# 3. サービス再起動
docker-compose restart

# 4. インデックス再構築
curl -X POST http://localhost:8000/api/management/reindex
```

### パフォーマンス調整

```bash
# 1. 環境変数の調整
nano .env

# 主要なパラメータ:
# MAX_WORKERS=4        # 並列処理数
# BATCH_SIZE=32        # バッチサイズ
# SEARCH_TOP_K=5       # 検索結果数
# CHUNK_SIZE=500       # チャンクサイズ

# 2. 設定の反映
docker-compose restart
```

## 監視とアラート

### Prometheus/Grafana連携（オプション）

```yaml
# docker-compose.monitoring.yml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

### カスタムメトリクス

```python
# src/utils/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# メトリクス定義
REQUEST_COUNT = Counter('rag_requests_total', 'Total requests')
REQUEST_DURATION = Histogram('rag_request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('rag_active_connections', 'Active connections')
```

## 運用チェックリスト

### 日次チェック

- [ ] システム状態確認
- [ ] ログエラー確認
- [ ] バックアップ実行確認
- [ ] リソース使用率確認
- [ ] API応答時間確認

### 週次チェック

- [ ] パフォーマンス分析
- [ ] データ品質チェック
- [ ] セキュリティ更新確認
- [ ] 容量使用状況確認

### 月次チェック

- [ ] モデル更新検討
- [ ] 設定最適化
- [ ] 障害対応手順見直し
- [ ] 運用ドキュメント更新

## 連絡先とエスカレーション

### 緊急時連絡先

- **システム管理者**: [連絡先]
- **技術サポート**: [連絡先]
- **ベンダーサポート**: [連絡先]

### エスカレーション基準

- **レベル1**: 軽微な問題（応答遅延、警告ログ）
- **レベル2**: 中程度の問題（一部機能停止、エラー増加）
- **レベル3**: 重大な問題（サービス全停止、データ破損）