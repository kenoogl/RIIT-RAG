# スーパーコンピュータ運用支援RAGシステム

九州大学情報基盤研究開発センターのスーパーコンピュータに関する質問応答を提供するRAG（Retrieval-Augmented Generation）システムです。

## プロジェクト構造

```
.
├── src/                    # ソースコード
│   ├── models/            # データモデル
│   ├── services/          # サービス層
│   ├── api/              # API層
│   └── utils/            # ユーティリティ
├── tests/                 # テストコード
├── data/                 # データファイル（実行時作成）
├── logs/                 # ログファイル（実行時作成）
├── models/               # AIモデル（実行時作成）
├── config.yaml           # 設定ファイル
├── requirements.txt      # 依存関係
├── main.py              # メインエントリーポイント
└── README.md            # このファイル
```

## セットアップ

1. 依存関係のインストール:
```bash
pip install -r requirements.txt
```

2. アプリケーションの実行:
```bash
python main.py
```

## 設定

`config.yaml`ファイルでシステムの動作を設定できます。主な設定項目：

- **app**: アプリケーション基本設定
- **crawler**: Webクローリング設定
- **document_processing**: 文書処理設定
- **embedding**: 埋め込みモデル設定
- **vector_db**: ベクトルデータベース設定
- **generation**: 生成モデル設定
- **search**: 検索設定
- **storage**: ストレージ設定
- **logging**: ログ設定
- **performance**: パフォーマンス設定

## 機能

- 九州大学SCPサイトからの文書自動収集
- 日本語文書の処理とインデックス化
- ベクトル検索による関連文書の検索
- ローカル生成AIモデルによる回答生成
- オンプレミス環境での完全動作

## 要件

- Python 3.8以上
- 十分なディスク容量（AIモデル保存用）
- メモリ4GB以上推奨