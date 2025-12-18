# API仕様書

## 概要

スーパーコンピュータ運用支援RAGシステムのREST API仕様書です。質問応答機能と管理機能のエンドポイントを提供します。

## ベースURL

```
http://localhost:8000
```

## 認証

現在のバージョンでは認証は実装されていません。本番環境では適切な認証機構の実装を推奨します。

## エンドポイント一覧

### 質問応答API

#### POST /api/question

ユーザーからの質問に対して回答を生成します。

**リクエスト**:
```json
{
  "question": "スパコンのアカウント作成方法を教えてください"
}
```

**レスポンス**:
```json
{
  "answer": "スパコンのアカウント作成は以下の手順で行います...",
  "sources": [
    {
      "title": "アカウント作成ガイド",
      "url": "https://www.cc.kyushu-u.ac.jp/scp/account/",
      "score": 0.85
    }
  ],
  "processing_time": 2.34,
  "confidence": 0.92
}
```

**ステータスコード**:
- `200`: 成功
- `400`: 不正なリクエスト
- `500`: サーバーエラー

#### GET /health

システムの健康状態を確認します。

**レスポンス**:
```json
{
  "status": "healthy",
  "timestamp": "2023-12-01T10:00:00Z",
  "version": "1.0.0",
  "components": {
    "database": "healthy",
    "embedding_model": "healthy",
    "generation_model": "healthy"
  }
}
```

### 管理API

#### GET /api/management/status

システムの詳細な状態情報を取得します。

**レスポンス**:
```json
{
  "system_status": "running",
  "document_count": 150,
  "index_count": 1250,
  "last_crawl": "2023-12-01T09:00:00Z",
  "memory_usage": {
    "used": "2.1GB",
    "total": "4.0GB",
    "percentage": 52.5
  },
  "disk_usage": {
    "used": "15.2GB",
    "total": "50.0GB",
    "percentage": 30.4
  }
}
```

#### POST /api/management/crawl

指定されたURLから文書を収集します。

**リクエスト**:
```json
{
  "url": "https://www.cc.kyushu-u.ac.jp/scp/",
  "max_depth": 3,
  "force_update": false
}
```

**レスポンス**:
```json
{
  "task_id": "crawl_20231201_100000",
  "status": "started",
  "message": "文書収集を開始しました"
}
```

#### POST /api/management/reindex

インデックスを再構築します。

**リクエスト**:
```json
{
  "force": false
}
```

**レスポンス**:
```json
{
  "task_id": "reindex_20231201_100000",
  "status": "started",
  "estimated_time": "10-15分"
}
```

#### GET /api/management/documents

管理されている文書の一覧を取得します。

**クエリパラメータ**:
- `page`: ページ番号（デフォルト: 1）
- `limit`: 1ページあたりの件数（デフォルト: 20）
- `search`: 検索キーワード（オプション）

**レスポンス**:
```json
{
  "documents": [
    {
      "id": "doc_001",
      "title": "スパコン利用ガイド",
      "url": "https://www.cc.kyushu-u.ac.jp/scp/guide/",
      "created_at": "2023-12-01T09:00:00Z",
      "updated_at": "2023-12-01T09:00:00Z",
      "chunk_count": 15,
      "language": "ja"
    }
  ],
  "total": 150,
  "page": 1,
  "limit": 20,
  "total_pages": 8
}
```

#### DELETE /api/management/documents/{document_id}

指定された文書を削除します。

**レスポンス**:
```json
{
  "message": "文書が正常に削除されました",
  "document_id": "doc_001"
}
```

#### GET /api/management/logs

システムログを取得します。

**クエリパラメータ**:
- `level`: ログレベル（DEBUG, INFO, WARNING, ERROR, CRITICAL）
- `limit`: 取得件数（デフォルト: 100）
- `since`: 開始日時（ISO 8601形式）

**レスポンス**:
```json
{
  "logs": [
    {
      "timestamp": "2023-12-01T10:00:00Z",
      "level": "INFO",
      "message": "文書処理が完了しました",
      "component": "document_engine"
    }
  ],
  "total": 1000,
  "limit": 100
}
```

## エラーレスポンス

すべてのエラーレスポンスは以下の形式で返されます：

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "リクエストが無効です",
    "details": "question フィールドが必要です"
  },
  "timestamp": "2023-12-01T10:00:00Z"
}
```

### エラーコード一覧

| コード | 説明 |
|--------|------|
| `INVALID_REQUEST` | リクエストが無効 |
| `QUESTION_TOO_LONG` | 質問が長すぎる |
| `NO_DOCUMENTS_FOUND` | 関連文書が見つからない |
| `MODEL_ERROR` | モデル処理エラー |
| `DATABASE_ERROR` | データベースエラー |
| `INTERNAL_ERROR` | 内部サーバーエラー |

## 使用例

### Python

```python
import requests

# 質問応答
response = requests.post(
    "http://localhost:8000/api/question",
    json={"question": "バッチジョブの投入方法は？"}
)
result = response.json()
print(result["answer"])

# システム状態確認
response = requests.get("http://localhost:8000/health")
status = response.json()
print(f"Status: {status['status']}")
```

### curl

```bash
# 質問応答
curl -X POST http://localhost:8000/api/question \
  -H "Content-Type: application/json" \
  -d '{"question": "スパコンの利用料金について教えて"}'

# 文書収集
curl -X POST http://localhost:8000/api/management/crawl \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.cc.kyushu-u.ac.jp/scp/"}'

# システム状態確認
curl http://localhost:8000/api/management/status
```

### JavaScript

```javascript
// 質問応答
const response = await fetch('http://localhost:8000/api/question', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    question: 'ジョブスケジューラの使い方は？'
  })
});

const result = await response.json();
console.log(result.answer);
```

## レート制限

現在のバージョンではレート制限は実装されていませんが、本番環境では以下の制限を推奨します：

- 質問応答API: 10リクエスト/分/IP
- 管理API: 100リクエスト/分/IP

## WebSocket API（将来実装予定）

リアルタイム通信のためのWebSocket APIを将来実装予定です：

```javascript
// WebSocket接続例
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  if (data.type === 'answer_chunk') {
    // ストリーミング回答の処理
    console.log(data.content);
  }
};

// 質問送信
ws.send(JSON.stringify({
  type: 'question',
  content: 'スパコンの使い方は？'
}));
```

## セキュリティ考慮事項

### 本番環境での推奨設定

1. **HTTPS の使用**
```nginx
server {
    listen 443 ssl;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
}
```

2. **API キー認証**
```python
# ヘッダーでのAPI キー認証
headers = {
    'Authorization': 'Bearer your-api-key',
    'Content-Type': 'application/json'
}
```

3. **CORS 設定**
```python
# 許可するオリジンの設定
ALLOWED_ORIGINS = [
    "https://your-domain.com",
    "https://admin.your-domain.com"
]
```

4. **入力検証**
- 質問の最大長: 1000文字
- SQLインジェクション対策
- XSS対策

## 監視とログ

### メトリクス

APIは以下のメトリクスを出力します：

- `rag_requests_total`: 総リクエスト数
- `rag_request_duration_seconds`: リクエスト処理時間
- `rag_active_connections`: アクティブ接続数
- `rag_model_inference_time`: モデル推論時間

### ログ形式

```json
{
  "timestamp": "2023-12-01T10:00:00Z",
  "level": "INFO",
  "component": "api",
  "endpoint": "/api/question",
  "method": "POST",
  "status_code": 200,
  "processing_time": 2.34,
  "user_ip": "192.168.1.100",
  "question_length": 25,
  "answer_length": 150
}
```