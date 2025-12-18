#!/usr/bin/env python3
"""
基本エンドツーエンドテスト - スーパーコンピュータ運用支援RAGシステム

システムの基本的な機能を検証するエンドツーエンドテスト
"""

import json
import logging
import time
import os
from datetime import datetime
from typing import Dict, List, Any
import sys

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BasicEndToEndTest:
    """基本エンドツーエンドテストクラス"""
    
    def __init__(self):
        """テスト初期化"""
        self.test_results = {
            "start_time": datetime.now().isoformat(),
            "tests": {},
            "summary": {}
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """全てのエンドツーエンドテストを実行"""
        logger.info("=== 基本エンドツーエンドテスト開始 ===")
        
        try:
            # 1. ファイル構造テスト
            self.test_file_structure()
            
            # 2. 設定ファイルテスト
            self.test_configuration()
            
            # 3. モジュールインポートテスト
            self.test_module_imports()
            
            # 4. データモデルテスト
            self.test_data_models()
            
            # 5. サービス初期化テスト
            self.test_service_initialization()
            
            # 6. 統合機能テスト
            self.test_integration_functionality()
            
            # テスト結果サマリー
            self.generate_test_summary()
            
        except Exception as e:
            logger.error(f"エンドツーエンドテスト中にエラーが発生: {e}")
            self.test_results["error"] = str(e)
        
        finally:
            self.test_results["end_time"] = datetime.now().isoformat()
            self.save_test_results()
        
        return self.test_results
    
    def test_file_structure(self):
        """ファイル構造テスト"""
        logger.info("--- ファイル構造テスト ---")
        test_name = "file_structure"
        
        try:
            # 必要なディレクトリの確認
            required_dirs = [
                "src", "src/api", "src/services", "src/models", "src/utils",
                "tests", "data", "logs", "models"
            ]
            
            # 重要なファイルの確認
            required_files = [
                "config.yaml", "requirements.txt", "README.md",
                "src/__init__.py", "src/models/core.py", "src/services/rag_service.py"
            ]
            
            dirs_exist = []
            files_exist = []
            
            for dir_path in required_dirs:
                exists = os.path.exists(dir_path)
                dirs_exist.append(exists)
                if not exists:
                    logger.warning(f"ディレクトリが見つかりません: {dir_path}")
            
            for file_path in required_files:
                exists = os.path.exists(file_path)
                files_exist.append(exists)
                if not exists:
                    logger.warning(f"ファイルが見つかりません: {file_path}")
            
            dirs_ok = all(dirs_exist)
            files_ok = all(files_exist)
            
            self.test_results["tests"][test_name] = {
                "status": "passed" if dirs_ok and files_ok else "failed",
                "directories_exist": dirs_ok,
                "files_exist": files_ok,
                "missing_directories": [d for d, exists in zip(required_dirs, dirs_exist) if not exists],
                "missing_files": [f for f, exists in zip(required_files, files_exist) if not exists]
            }
            
            logger.info(f"✓ ファイル構造テスト: {'成功' if dirs_ok and files_ok else '失敗'}")
            
        except Exception as e:
            self.test_results["tests"][test_name] = {
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"ファイル構造テストでエラー: {e}")
    
    def test_configuration(self):
        """設定ファイルテスト"""
        logger.info("--- 設定ファイルテスト ---")
        test_name = "configuration"
        
        try:
            # config.yamlの読み込みテスト
            config_exists = os.path.exists("config.yaml")
            
            if config_exists:
                try:
                    import yaml
                    with open("config.yaml", 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)
                    config_valid = isinstance(config_data, dict) and len(config_data) > 0
                    config_sections = list(config_data.keys()) if config_valid else []
                except Exception as e:
                    config_valid = False
                    config_sections = []
                    logger.error(f"設定ファイル読み込みエラー: {e}")
            else:
                config_valid = False
                config_sections = []
            
            self.test_results["tests"][test_name] = {
                "status": "passed" if config_exists and config_valid else "failed",
                "config_exists": config_exists,
                "config_valid": config_valid,
                "config_sections": config_sections
            }
            
            logger.info(f"✓ 設定ファイルテスト: {'成功' if config_exists and config_valid else '失敗'}")
            
        except Exception as e:
            self.test_results["tests"][test_name] = {
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"設定ファイルテストでエラー: {e}")
    
    def test_module_imports(self):
        """モジュールインポートテスト"""
        logger.info("--- モジュールインポートテスト ---")
        test_name = "module_imports"
        
        try:
            import_results = {}
            
            # 重要なモジュールのインポートテスト
            modules_to_test = [
                "src.models.core",
                "src.services.rag_service",
                "src.services.document_engine",
                "src.services.vector_database",
                "src.services.embedding_model",
                "src.utils.config"
            ]
            
            for module_name in modules_to_test:
                try:
                    __import__(module_name)
                    import_results[module_name] = True
                    logger.info(f"  ✓ {module_name}")
                except Exception as e:
                    import_results[module_name] = False
                    logger.error(f"  ✗ {module_name}: {e}")
            
            success_count = sum(import_results.values())
            success_rate = (success_count / len(modules_to_test)) * 100
            
            self.test_results["tests"][test_name] = {
                "status": "passed" if success_rate >= 80 else "failed",
                "success_rate": success_rate,
                "successful_imports": success_count,
                "total_imports": len(modules_to_test),
                "import_results": import_results
            }
            
            logger.info(f"✓ モジュールインポートテスト: {success_rate:.1f}% 成功率")
            
        except Exception as e:
            self.test_results["tests"][test_name] = {
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"モジュールインポートテストでエラー: {e}")
    
    def test_data_models(self):
        """データモデルテスト"""
        logger.info("--- データモデルテスト ---")
        test_name = "data_models"
        
        try:
            from src.models.core import Document, Chunk, SearchResult, Answer
            
            # データモデルのインスタンス作成テスト
            model_tests = {}
            
            # Document モデルテスト
            try:
                doc = Document(
                    id="test_001",
                    url="https://example.com",
                    title="テスト文書",
                    content="テスト内容",
                    language="ja"
                )
                model_tests["Document"] = True
            except Exception as e:
                model_tests["Document"] = False
                logger.error(f"Document モデルエラー: {e}")
            
            # Chunk モデルテスト
            try:
                chunk = Chunk(
                    id="chunk_001",
                    document_id="test_001",
                    content="チャンク内容",
                    position=0
                )
                model_tests["Chunk"] = True
            except Exception as e:
                model_tests["Chunk"] = False
                logger.error(f"Chunk モデルエラー: {e}")
            
            # SearchResult モデルテスト
            try:
                if model_tests.get("Document") and model_tests.get("Chunk"):
                    search_result = SearchResult(
                        chunk=chunk,
                        score=0.85,
                        document=doc
                    )
                    model_tests["SearchResult"] = True
                else:
                    model_tests["SearchResult"] = False
            except Exception as e:
                model_tests["SearchResult"] = False
                logger.error(f"SearchResult モデルエラー: {e}")
            
            # Answer モデルテスト
            try:
                answer = Answer(
                    text="テスト回答",
                    sources=["source1", "source2"],
                    confidence=0.9,
                    processing_time=1.5
                )
                model_tests["Answer"] = True
            except Exception as e:
                model_tests["Answer"] = False
                logger.error(f"Answer モデルエラー: {e}")
            
            success_count = sum(model_tests.values())
            success_rate = (success_count / len(model_tests)) * 100
            
            self.test_results["tests"][test_name] = {
                "status": "passed" if success_rate >= 75 else "failed",
                "success_rate": success_rate,
                "successful_models": success_count,
                "total_models": len(model_tests),
                "model_tests": model_tests
            }
            
            logger.info(f"✓ データモデルテスト: {success_rate:.1f}% 成功率")
            
        except Exception as e:
            self.test_results["tests"][test_name] = {
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"データモデルテストでエラー: {e}")
    
    def test_service_initialization(self):
        """サービス初期化テスト"""
        logger.info("--- サービス初期化テスト ---")
        test_name = "service_initialization"
        
        try:
            service_tests = {}
            
            # VectorDatabase 初期化テスト
            try:
                from src.services.vector_database import LocalVectorDatabase
                vector_db = LocalVectorDatabase()
                service_tests["LocalVectorDatabase"] = True
            except Exception as e:
                service_tests["LocalVectorDatabase"] = False
                logger.error(f"LocalVectorDatabase 初期化エラー: {e}")
            
            # EmbeddingModel 初期化テスト
            try:
                from src.services.embedding_model import LocalEmbeddingModel
                embedding_model = LocalEmbeddingModel()
                service_tests["LocalEmbeddingModel"] = True
            except Exception as e:
                service_tests["LocalEmbeddingModel"] = False
                logger.error(f"LocalEmbeddingModel 初期化エラー: {e}")
            
            # DocumentEngine 初期化テスト
            try:
                from src.services.document_engine import DocumentProcessingEngine
                doc_engine = DocumentProcessingEngine()
                service_tests["DocumentProcessingEngine"] = True
            except Exception as e:
                service_tests["DocumentProcessingEngine"] = False
                logger.error(f"DocumentProcessingEngine 初期化エラー: {e}")
            
            # ManagementService 初期化テスト
            try:
                from src.services.management_service import DocumentManagementService
                mgmt_service = DocumentManagementService()
                service_tests["DocumentManagementService"] = True
            except Exception as e:
                service_tests["DocumentManagementService"] = False
                logger.error(f"DocumentManagementService 初期化エラー: {e}")
            
            success_count = sum(service_tests.values())
            success_rate = (success_count / len(service_tests)) * 100
            
            self.test_results["tests"][test_name] = {
                "status": "passed" if success_rate >= 75 else "failed",
                "success_rate": success_rate,
                "successful_services": success_count,
                "total_services": len(service_tests),
                "service_tests": service_tests
            }
            
            logger.info(f"✓ サービス初期化テスト: {success_rate:.1f}% 成功率")
            
        except Exception as e:
            self.test_results["tests"][test_name] = {
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"サービス初期化テストでエラー: {e}")
    
    def test_integration_functionality(self):
        """統合機能テスト"""
        logger.info("--- 統合機能テスト ---")
        test_name = "integration_functionality"
        
        try:
            integration_tests = {}
            
            # 設定読み込み機能テスト
            try:
                from src.utils.config import load_config
                config = load_config()
                integration_tests["config_loading"] = config is not None
            except Exception as e:
                integration_tests["config_loading"] = False
                logger.error(f"設定読み込みエラー: {e}")
            
            # 文書管理機能テスト
            try:
                from src.services.management_service import DocumentManagementService
                mgmt = DocumentManagementService()
                documents = mgmt.list_documents()
                integration_tests["document_management"] = isinstance(documents, list)
            except Exception as e:
                integration_tests["document_management"] = False
                logger.error(f"文書管理エラー: {e}")
            
            # システム監視機能テスト
            try:
                from src.services.management_service import SystemMonitoringService
                monitor = SystemMonitoringService()
                status = monitor.get_system_status()
                integration_tests["system_monitoring"] = isinstance(status, dict)
            except Exception as e:
                integration_tests["system_monitoring"] = False
                logger.error(f"システム監視エラー: {e}")
            
            # 既存テストファイルの実行確認
            try:
                test_files = [
                    "integration_test.py",
                    "system_integration_check.py",
                    "verify_offline.py"
                ]
                existing_tests = [f for f in test_files if os.path.exists(f)]
                integration_tests["existing_test_files"] = len(existing_tests) >= 2
            except Exception as e:
                integration_tests["existing_test_files"] = False
                logger.error(f"既存テストファイル確認エラー: {e}")
            
            success_count = sum(integration_tests.values())
            success_rate = (success_count / len(integration_tests)) * 100
            
            self.test_results["tests"][test_name] = {
                "status": "passed" if success_rate >= 75 else "failed",
                "success_rate": success_rate,
                "successful_integrations": success_count,
                "total_integrations": len(integration_tests),
                "integration_tests": integration_tests
            }
            
            logger.info(f"✓ 統合機能テスト: {success_rate:.1f}% 成功率")
            
        except Exception as e:
            self.test_results["tests"][test_name] = {
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"統合機能テストでエラー: {e}")
    
    def generate_test_summary(self):
        """テスト結果のサマリーを生成"""
        tests = self.test_results["tests"]
        total_tests = len(tests)
        passed_tests = sum(1 for test in tests.values() if test.get("status") == "passed")
        
        self.test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "overall_status": "PASSED" if passed_tests >= total_tests * 0.8 else "FAILED"
        }
        
        logger.info(f"=== テスト結果サマリー ===")
        logger.info(f"総テスト数: {total_tests}")
        logger.info(f"成功: {passed_tests}")
        logger.info(f"失敗: {total_tests - passed_tests}")
        logger.info(f"成功率: {self.test_results['summary']['success_rate']:.1f}%")
        logger.info(f"総合結果: {self.test_results['summary']['overall_status']}")
    
    def save_test_results(self):
        """テスト結果をファイルに保存"""
        results_file = "basic_e2e_test_results.json"
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, ensure_ascii=False, indent=2)
            logger.info(f"テスト結果を保存しました: {results_file}")
        except Exception as e:
            logger.error(f"テスト結果の保存に失敗: {e}")

def main():
    """メイン実行関数"""
    test_runner = BasicEndToEndTest()
    results = test_runner.run_all_tests()
    
    # 結果の表示
    print("\n" + "="*50)
    print("基本エンドツーエンドテスト完了")
    print("="*50)
    print(f"総合結果: {results['summary']['overall_status']}")
    print(f"成功率: {results['summary']['success_rate']:.1f}%")
    print(f"詳細結果: basic_e2e_test_results.json")
    
    # 終了コード
    return 0 if results['summary']['overall_status'] == 'PASSED' else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)