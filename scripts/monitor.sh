#!/bin/bash
# スーパーコンピュータ運用支援RAGシステム - 監視スクリプト

set -e

echo "=== RAGシステム 監視ダッシュボード ==="

# システム状態の確認
check_system_status() {
    echo "--- システム状態 ---"
    
    # Dockerコンテナの状態
    if command -v docker-compose &> /dev/null; then
        echo "Dockerコンテナ状態:"
        docker-compose ps 2>/dev/null || echo "  Docker Composeが実行されていません"
    fi
    
    # プロセスの確認
    echo "RAGシステムプロセス:"
    pgrep -f "run_api.py" > /dev/null && echo "  ✓ APIサーバーが実行中" || echo "  ❌ APIサーバーが停止中"
    
    # ポートの確認
    echo "ポート使用状況:"
    netstat -tlnp 2>/dev/null | grep ":8000" > /dev/null && echo "  ✓ ポート8000が使用中" || echo "  ❌ ポート8000が未使用"
}

# リソース使用状況の確認
check_resources() {
    echo "--- リソース使用状況 ---"
    
    # CPU使用率
    echo "CPU使用率:"
    top -bn1 | grep "Cpu(s)" | awk '{print "  " $2}' || echo "  取得できませんでした"
    
    # メモリ使用率
    echo "メモリ使用率:"
    free -h | grep "Mem:" | awk '{print "  使用中: " $3 " / 合計: " $2}' || echo "  取得できませんでした"
    
    # ディスク使用率
    echo "ディスク使用率:"
    df -h . | tail -1 | awk '{print "  使用中: " $3 " / 合計: " $2 " (" $5 ")"}' || echo "  取得できませんでした"
}

# ログの確認
check_logs() {
    echo "--- 最新ログ ---"
    
    # エラーログの確認
    if [ -f "logs/rag_system.log" ]; then
        echo "最新のエラー（直近10件）:"
        tail -n 100 logs/rag_system.log | grep -i "error\|exception\|failed" | tail -10 || echo "  エラーはありません"
    else
        echo "  ログファイルが見つかりません"
    fi
}

# APIヘルスチェック
check_api_health() {
    echo "--- APIヘルスチェック ---"
    
    if command -v curl &> /dev/null; then
        response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null || echo "000")
        if [ "$response" = "200" ]; then
            echo "  ✓ APIが正常に応答しています"
        else
            echo "  ❌ APIが応答していません (HTTP: $response)"
        fi
    else
        echo "  curlが利用できません"
    fi
}

# データファイルの確認
check_data_files() {
    echo "--- データファイル状況 ---"
    
    # 文書数の確認
    if [ -d "data/documents" ]; then
        doc_count=$(find data/documents -name "*.json" | wc -l)
        echo "  文書数: $doc_count"
    else
        echo "  文書ディレクトリが見つかりません"
    fi
    
    # ベクトルファイルの確認
    if [ -d "data/vectors" ]; then
        vector_files=$(find data/vectors -name "*.npy" | wc -l)
        echo "  ベクトルファイル数: $vector_files"
    else
        echo "  ベクトルディレクトリが見つかりません"
    fi
}

# メイン実行
main() {
    while true; do
        clear
        echo "$(date '+%Y-%m-%d %H:%M:%S') - RAGシステム監視"
        echo "=================================================="
        
        check_system_status
        echo ""
        check_resources
        echo ""
        check_logs
        echo ""
        check_api_health
        echo ""
        check_data_files
        
        echo ""
        echo "=================================================="
        echo "更新間隔: 30秒 (Ctrl+Cで終了)"
        
        sleep 30
    done
}

# 引数に応じて実行
case "${1:-monitor}" in
    "status")
        check_system_status
        ;;
    "resources")
        check_resources
        ;;
    "logs")
        check_logs
        ;;
    "health")
        check_api_health
        ;;
    "data")
        check_data_files
        ;;
    "monitor"|"")
        main
        ;;
    *)
        echo "使用方法: $0 [status|resources|logs|health|data|monitor]"
        exit 1
        ;;
esac