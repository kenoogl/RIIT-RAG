#!/usr/bin/env python3
"""
簡易エンドツーエンドテスト - スーパーコンピュータ運用支援RAGシステム

生成モデルなしでも実行可能な基本的なエンドツーエンドテストを実行する。
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import load_config
from src.services.web_crawler import RobustWebCrawler
from src.services.document_engine import DocumentProcessingEngine
from src.services.vector_database import LocalVectorDatabase
from src.services.embedding_model import LocalEmbeddingModel
from src.services.enhanced_search_engine import EnhancedSearchEngine
from src.services.management_service import DocumentManagementService, SystemMonitoringService

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleEndToEndTest:
    """簡易エンドツーエンドテストクラス"""
    
    def __init__(self):
        """テスト初期化"""
        self.config = load_config()
        self.test_results = {
            "start_time": datetime.now().isoformat(),
            "tests": {},
            "summary": {}
        }
        
        # テスト用質問セット
        self.test_questions = [
            "スパコンのアカウント作成方法を教えてください",
            "バッチジョブの投入方法は？",
            "利用料金について知りたい"
        ]
    
    def run_all_tests(self) -> Dict[str, Any]:
        """全てのエンドツーエンドテストを実行"""
        logger.info("=== 簡易エンドツーエンドテスト開始 ===")
        
        try:
            # 1. システム初期化テスト
            self.test_system_initialization()
            
            # 2. 文書収集・処理テスト
            self.test_document_collection_and_processing()
            
            # 3. 検索機能テスト
            self.test_search_functionality()
            
            # 4. 管理機能テスト
            self.test_management_functions()
            
            # 5. データ整合性テスト
            self.test_data_integrity()
            
            # テスト結果サマリー
            self.generate_test_summary()
            
        except Exception as e:
            logger.error(f"エンドツーエンドテスト中にエラーが発生: {e}")
            self.test_results["error"] = str(e)
        
        finally:
            self.test_results["end_time"] = datetime.now().isoformat()
            self.save_test_results()
        
        return self.test_results
    
    def test_system_initialization(self):
        """システム初期化テスト"""
        logger.info("--- システム初期化テスト ---")
        test_name = "system_initialization"
        
        try:
            # 設定ファイルの読み込み確認
            config_loaded = self.config is not None
            
            # 必要なディレクトリの存在確認
            required_dirs = ["data", "logs", "models"]
            dirs_exist = all(os.path.exists(d) for d in required_dirs)
            
            # サービスの初期化確認
            try:
                vector_db = LocalVectorDatabase()
                embedding_model = LocalEmbeddingModel()
                search_engine = EnhancedSearchEngine(vector_db, embedding_model)
                services_init = True
            except Exception as e:
                logger.error(f"サービス初期化エラー: {e}")
                services_init = False
            
            self.test_results["tests"][test_name] = {
                "status": "passed" if all([config_loaded, dirs_exist, services_init]) else "failed",
                "config_loaded": config_loaded,
                "directories_exist": dirs_exist,
                "services_initialized": services_init,
                "details": {
                    "config_sections": list(self.config.keys()) if config_loaded else None
                }
            }
            
            logger.info(f"✓ システム初期化テスト: {'成功' if self.test_results['tests'][test_name]['status'] == 'passed' else '失敗'}")
            
        except Exception as e:
            self.test_results["tests"][test_name] = {
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"システム初期化テストでエラー: {e}")
    
    def test_document_collection_and_processing(self):
        """文書収集・処理テスト"""
        logger.info("--- 文書収集・処理テスト ---")
        test_name = "document_collection_processing"
        
        try:
            # 文書処理エンジンの初期化
            doc_engine = DocumentProcessingEngine()
            
            # テスト用文書の作成
            test_document = {
                "id": "test_doc_001",
                "url": "https://example.com/test",
                "title": "テスト文書",
                "content": "これはスーパーコンピュータのテスト文書です。アカウント作成やバッチジョブの投入方法について説明します。",
                "language": "ja"
            }
            
            # 文書処理の実行
            start_time = time.time()
            processing_result = doc_engine.process_document_data(test_document)
            processing_time = time.time() - start_time
            
            # 処理結果の確認
            chunks_created = len(processing_result.get("chunks", []))
            
            self.test_results["tests"][test_name] = {
                "status": "passed" if chunks_created > 0 else "failed",
                "processing_time": processing_time,
                "chunks_created": chunks_created,
                "processing_result": processing_result
            }
            
            logger.info(f"✓ 文書収集・処理テスト: {chunks_created}個のチャンクを作成")
            
        except Exception as e:
            self.test_results["tests"][test_name] = {
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"文書収集・処理テストでエラー: {e}")
    
    def test_search_functionality(self):
        """検索機能テスト"""
        logger.info("--- 検索機能テスト ---")
        test_name = "search_functionality"
        
        try:
            # 検索エンジンの初期化
            vector_db = LocalVectorDatabase()
            embedding_model = LocalEmbeddingModel()
            search_engine = EnhancedSearchEngine(vector_db, embedding_model)
            
            search_results = []
            
            for question in self.test_questions:
                logger.info(f"検索テスト: {question}")
                
                start_time = time.time()
                try:
                    # 検索の実行
                    results = search_engine.search(question, top_k=3)
                    search_time = time.time() - start_time
                    
                    search_results.append({
                        "question": question,
                        "search_time": search_time,
                        "results_count": len(results),
                        "success": True
                    })
                    
                    logger.info(f"  検索時間: {search_time:.2f}秒, 結果数: {len(results)}")
                    
                except Exception as e:
                    search_results.append({
                        "question": question,
                        "error": str(e),
                        "success": False
                    })
                    logger.error(f"  検索エラー: {e}")
            
            # 成功率の計算
            successful_searches = sum(1 for r in search_results if r.get("success", False))
            success_rate = (successful_searches / len(search_results)) * 100
            
            self.test_results["tests"][test_name] = {
                "status": "passed" if success_rate >= 50 else "failed",
                "success_rate": success_rate,
                "total_searches": len(search_results),
                "successful_searches": successful_searches,
                "search_results": search_results
            }
            
            logger.info(f"✓ 検索機能テスト: {success_rate:.1f}% 成功率")
            
        except Exception as e:
            self.test_results["tests"][test_name] = {
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"検索機能テストでエラー: {e}")
    
    def test_management_functions(self):
        """管理機能テスト"""
        logger.info("--- 管理機能テスト ---")
        test_name = "management_functions"
        
        try:
            # 管理サービスの初期化
            doc_mgmt = DocumentManagementService()
            sys_monitor = SystemMonitoringService()
            
            # 文書管理機能のテスト
            documents = doc_mgmt.list_documents()
            doc_mgmt_ok = isinstance(documents, list)
            
            # システム監視機能のテスト
            status = sys_monitor.get_system_status()
            sys_monitor_ok = isinstance(status, dict)
            
            # インデックス状態の確認
            index_status = doc_mgmt.get_index_status()
            index_ok = isinstance(index_status, dict)
            
            self.test_results["tests"][test_name] = {
                "status": "passed" if all([doc_mgmt_ok, sys_monitor_ok, index_ok]) else "failed",
                "document_management": doc_mgmt_ok,
                "system_monitoring": sys_monitor_ok,
                "index_status": index_ok,
                "documents_count": len(documents) if doc_mgmt_ok else 0,
                "system_status": status if sys_monitor_ok else None
            }
            
            logger.info(f"✓ 管理機能テスト: {'成功' if self.test_results['tests'][test_name]['status'] == 'passed' else '失敗'}")
            
        except Exception as e:
            self.test_results["tests"][test_name] = {
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"管理機能テストでエラー: {e}")
    
    def test_data_integrity(self):
        """データ整合性テスト"""
        logger.info("--- データ整合性テスト ---")
        test_name = "data_integrity"
        
        try:
            # ベクトルデータベースの整合性確認
            vector_db = LocalVectorDatabase()
            
            # データディレクトリの確認
            data_dirs = ["data/documents", "data/vectors", "data/logs"]
            dirs_accessible = all(os.path.exists(d) for d in data_dirs)
            
            # 設定ファイルの整合性確認
            config_valid = self.config is not None and len(self.config) > 0
            
            # ログファイルの確認
            log_files_exist = os.path.exists("logs") and len(os.listdir("logs")) >= 0
            
            integrity_ok = all([dirs_accessible, config_valid, log_files_exist])
            
            self.test_results["tests"][test_name] = {
                "status": "passed" if integrity_ok else "failed",
                "data_directories_accessible": dirs_accessible,
                "config_valid": config_valid,
                "log_files_exist": log_files_exist,
                "overall_integrity": integrity_ok
            }
            
            logger.info(f"✓ データ整合性テスト: {'成功' if integrity_ok else '失敗'}")
            
        except Exception as e:
            self.test_results["tests"][test_name] = {
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"データ整合性テストでエラー: {e}")
    
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
            "overall_status": "PASSED" if passed_tests == total_tests else "FAILED"
        }
        
        logger.info(f"=== テスト結果サマリー ===")
        logger.info(f"総テスト数: {total_tests}")
        logger.info(f"成功: {passed_tests}")
        logger.info(f"失敗: {total_tests - passed_tests}")
        logger.info(f"成功率: {self.test_results['summary']['success_rate']:.1f}%")
        logger.info(f"総合結果: {self.test_results['summary']['overall_status']}")
    
    def save_test_results(self):
        """テスト結果をファイルに保存"""
        results_file = "simple_e2e_test_results.json"
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, ensure_ascii=False, indent=2)
            logger.info(f"テスト結果を保存しました: {results_file}")
        except Exception as e:
            logger.error(f"テスト結果の保存に失敗: {e}")

def main():
    """メイン実行関数"""
    test_runner = SimpleEndToEndTest()
    results = test_runner.run_all_tests()
    
    # 結果の表示
    print("\n" + "="*50)
    print("簡易エンドツーエンドテスト完了")
    print("="*50)
    print(f"総合結果: {results['summary']['overall_status']}")
    print(f"成功率: {results['summary']['success_rate']:.1f}%")
    print(f"詳細結果: simple_e2e_test_results.json")
    
    # 終了コード
    return 0 if results['summary']['overall_status'] == 'PASSED' else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)