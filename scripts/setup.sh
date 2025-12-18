#!/bin/bash
# スーパーコンピュータ運用支援RAGシステム - セットアップスクリプト

set -e

echo "=== スーパーコンピュータ運用支援RAGシステム セットアップ ==="

# 環境変数の設定
if [ ! -f .env ]; then
    echo "環境設定ファイルを作成中..."
    cp .env.example .env
    echo "✓ .env ファイルが作成されました。必要に応じて編集してください。"
fi

# 必要なディレクトリの作成
echo "データディレクトリを作成中..."
mkdir -p data/{documents,vectors,logs,processing,website_structure}
mkdir -p logs
mkdir -p models/{embedding,generation}
echo "✓ ディレクトリが作成されました。"

# Pythonの仮想環境作成（Docker使用時は不要）
if [ "$1" != "--docker" ]; then
    echo "Python仮想環境を作成中..."
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
        echo "✓ 仮想環境が作成されました。"
    fi
    
    echo "依存関係をインストール中..."
    source .venv/bin/activate
    pip install -r requirements.txt
    echo "✓ 依存関係がインストールされました。"
fi

# 設定ファイルの確認
if [ ! -f config.yaml ]; then
    echo "❌ config.yaml が見つかりません。"
    exit 1
fi

echo "✓ セットアップが完了しました。"

if [ "$1" = "--docker" ]; then
    echo ""
    echo "Dockerでの起動方法:"
    echo "  docker-compose up -d"
    echo ""
    echo "ログの確認:"
    echo "  docker-compose logs -f"
else
    echo ""
    echo "システムの起動方法:"
    echo "  source .venv/bin/activate"
    echo "  python run_api.py"
    echo ""
    echo "または管理インターフェース:"
    echo "  python demo_management.py"
fi