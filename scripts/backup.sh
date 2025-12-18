#!/bin/bash
# スーパーコンピュータ運用支援RAGシステム - バックアップスクリプト

set -e

echo "=== RAGシステム バックアップ ==="

# バックアップディレクトリの作成
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "バックアップを作成中: $BACKUP_DIR"

# データファイルのバックアップ
echo "データファイルをバックアップ中..."
if [ -d "data" ]; then
    cp -r data "$BACKUP_DIR/"
    echo "✓ データファイルをバックアップしました。"
fi

# ログファイルのバックアップ
echo "ログファイルをバックアップ中..."
if [ -d "logs" ]; then
    cp -r logs "$BACKUP_DIR/"
    echo "✓ ログファイルをバックアップしました。"
fi

# 設定ファイルのバックアップ
echo "設定ファイルをバックアップ中..."
cp config.yaml "$BACKUP_DIR/" 2>/dev/null || echo "⚠ config.yaml が見つかりません。"
cp .env "$BACKUP_DIR/" 2>/dev/null || echo "⚠ .env が見つかりません。"

# バックアップの圧縮
echo "バックアップを圧縮中..."
tar -czf "${BACKUP_DIR}.tar.gz" -C backups "$(basename "$BACKUP_DIR")"
rm -rf "$BACKUP_DIR"

echo "✓ バックアップが完了しました: ${BACKUP_DIR}.tar.gz"

# 古いバックアップの削除（30日以上前）
echo "古いバックアップを削除中..."
find backups -name "*.tar.gz" -mtime +30 -delete 2>/dev/null || true
echo "✓ 古いバックアップを削除しました。"