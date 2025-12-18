#!/bin/bash
# スーパーコンピュータ運用支援RAGシステム - デプロイメントスクリプト

set -e

echo "=== RAGシステム デプロイメント ==="

# 引数の確認
if [ $# -eq 0 ]; then
    echo "使用方法: $0 [dev|prod]"
    exit 1
fi

ENVIRONMENT=$1

# 環境別設定
case $ENVIRONMENT in
    "dev")
        echo "開発環境にデプロイ中..."
        COMPOSE_FILE="docker-compose.yml"
        ;;
    "prod")
        echo "本番環境にデプロイ中..."
        COMPOSE_FILE="docker-compose.prod.yml"
        if [ ! -f "$COMPOSE_FILE" ]; then
            echo "❌ $COMPOSE_FILE が見つかりません。"
            exit 1
        fi
        ;;
    *)
        echo "❌ 無効な環境: $ENVIRONMENT"
        echo "使用方法: $0 [dev|prod]"
        exit 1
        ;;
esac

# 既存のコンテナを停止
echo "既存のコンテナを停止中..."
docker-compose -f $COMPOSE_FILE down

# イメージをビルド
echo "Dockerイメージをビルド中..."
docker-compose -f $COMPOSE_FILE build

# コンテナを起動
echo "コンテナを起動中..."
docker-compose -f $COMPOSE_FILE up -d

# ヘルスチェック
echo "ヘルスチェック中..."
sleep 10

if docker-compose -f $COMPOSE_FILE ps | grep -q "Up"; then
    echo "✓ デプロイメントが完了しました。"
    echo ""
    echo "システム状態の確認:"
    echo "  docker-compose -f $COMPOSE_FILE ps"
    echo ""
    echo "ログの確認:"
    echo "  docker-compose -f $COMPOSE_FILE logs -f"
    echo ""
    echo "APIエンドポイント:"
    echo "  http://localhost:8000/health"
else
    echo "❌ デプロイメントに失敗しました。"
    echo "ログを確認してください:"
    echo "  docker-compose -f $COMPOSE_FILE logs"
    exit 1
fi