#!/usr/bin/env python3
"""
エンドツーエンドテスト - スーパーコンピュータ運用支援RAGシステム

実際のWebサイトを使用して、質問から回答までの全フローを確認する。
要件の全てを満たすことを検証する。
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any
import requests
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import load_config
from src.services.web_crawler import RobustWebCrawler
from src.services.document_engine import DocumentEngine
from src.services.rag_service import RAGService
from src.services.management_service import DocumentManagementService, SystemMonitoringService

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EndToEndTest:
    """エンドツーエンドテストクラス"""
    
    def __init__(self):
        """テスト初期化"""
        self.config = load_config()
        self.test_results = {
            "start_time": datetime.now().isoformat(),
            "tests": {},
            "summary": {}
        }
        self.api_base_url = "http://localhost:8000"
        
        # テスト用質問セット
        self.test_questions = [
            {
                "question": "スパコンのアカウント作成方法を教えてください",
                "expected_keywords": ["アカウント", "作成", "申請", "登録"],
                "requirement": "1.1, 1.2, 1.4"
            },
            {
                "question": "バッチジョブの投入方法は？",
                "expected_keywords": ["バッチ", "ジョブ", "投入", "qsub", "sbatch"],
                "requirement": "1.1, 1.2, 1.5"
            },
            {
                "question": "利用料金について知りたい",
                "expected_keywords": ["料金", "費用", "課金", "コスト"],
                "requirement": "1.1, 1.2, 1.4"
            },
            {
                "question": "存在しない情報について教えて",
                "expected_keywords": [],
                "requirement": "1.3",
                "expect_no_info": True
            }
        ]
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """全てのエンドツーエンドテストを実行"""
        logger.info("=== エンドツーエンドテスト開始 ===")
        
        try:
            # 1. システム初期化テスト
            await self.test_system_initialization()
            
            # 2. 文書収集・処理テスト
            await self.test_document_collection_and_processing()
            
            # 3. 質問応答フローテスト
            await self.test_question_answering_flow()
            
            # 4. 管理機能テスト
            await self.test_management_functions()
            
            # 5. エラーハンドリングテスト
            await self.test_error_handling()
            
            # 6. パフォーマンステスト
            await self.test_performance()
            
            # 7. オフライン動作テスト
            await self.test_offline_operation()
            
            # テスト結果サマリー
            self.generate_test_summary()
            
        except Exception as e:
            logger.error(f"エンドツーエンドテスト中にエラーが発生: {e}")
            self.test_results["error"] = str(e)
        
        finally:
            self.test_results["end_time"] = datetime.now().isoformat()
            self.save_test_results()
        
        return self.test_results
    
    async def test_system_initialization(self):
        """システム初期化テスト"""
        logger.info("--- システム初期化テスト ---")
        test_name = "system_initialization"
        
        try:
            # APIサーバーの起動確認
            response = requests.get(f"{self.api_base_url}/health", timeout=10)
            api_healthy = response.status_code == 200
            
            # 設定ファイルの読み込み確認
            config_loaded = self.config is not None
            
            # 必要なディレクトリの存在確認
            required_dirs = ["data", "logs", "models"]
            dirs_exist = all(os.path.exists(d) for d in required_dirs)
            
            self.test_results["tests"][test_name] = {
                "status": "passed" if all([api_healthy, config_loaded, dirs_exist]) else "failed",
                "api_healthy": api_healthy,
                "config_loaded": config_loaded,
                "directories_exist": dirs_exist,
                "details": {
                    "api_response": response.json() if api_healthy else None,
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
    
    async def test_document_collection_and_processing(self):
        """文書収集・処理テスト"""
        logger.info("--- 文書収集・処理テスト ---")
        test_name = "document_collection_processing"
        
        try:
            # 管理サービスを使用して文書収集を実行
            doc_mgmt = DocumentManagementService()
            
            # テスト用URL（実際のサイトの一部）
            test_url = "https://www.cc.kyushu-u.ac.jp/scp/"
            
            # 文書収集の実行
            start_time = time.time()
            collection_result = await self.trigger_document_collection(test_url)
            collection_time = time.time() - start_time
            
            # 処理結果の確認
            documents = doc_mgmt.list_documents()
            doc_count = len(documents)
            
            # インデックス状態の確認
            index_status = doc_mgmt.get_index_status()
            
            self.test_results["tests"][test_name] = {
                "status": "passed" if doc_count > 0 else "failed",
                "collection_time": collection_time,
                "documents_collected": doc_count,
                "index_status": index_status,
                "collection_result": collection_result,
                "sample_documents": documents[:3] if documents else []
            }
            
            logger.info(f"✓ 文書収集・処理テスト: {doc_count}件の文書を収集")
            
        except Exception as e:
            self.test_results["tests"][test_name] = {
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"文書収集・処理テストでエラー: {e}")
    
    async def test_question_answering_flow(self):
        """質問応答フローテスト"""
        logger.info("--- 質問応答フローテスト ---")
        test_name = "question_answering_flow"
        
        try:
            qa_results = []
            
            for i, test_case in enumerate(self.test_questions):
                logger.info(f"質問 {i+1}: {test_case['question']}")
                
                # 質問応答APIを呼び出し
                start_time = time.time()
                response = requests.post(
                    f"{self.api_base_url}/api/question",
                    json={"question": test_case["question"]},
                    timeout=30
                )
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    answer_data = response.json()
                    
                    # 回答の検証
                    validation_result = self.validate_answer(answer_data, test_case)
                    
                    qa_results.append({
                        "question": test_case["question"],
                        "response_time": response_time,
                        "answer": answer_data.get("answer", ""),
                        "sources": answer_data.get("sources", []),
                        "confidence": answer_data.get("confidence", 0),
                        "validation": validation_result,
                        "requirement": test_case["requirement"]
                    })
                    
                    logger.info(f"  回答時間: {response_time:.2f}秒")
                    logger.info(f"  検証結果: {'✓' if validation_result['passed'] else '✗'}")
                    
                else:
                    qa_results.append({
                        "question": test_case["question"],
                        "error": f"HTTP {response.status_code}",
                        "requirement": test_case["requirement"]
                    })
                
                # 次の質問まで少し待機
                await asyncio.sleep(1)
            
            # 全体の成功率を計算
            passed_count = sum(1 for r in qa_results if r.get("validation", {}).get("passed", False))
            success_rate = (passed_count / len(qa_results)) * 100
            
            self.test_results["tests"][test_name] = {
                "status": "passed" if success_rate >= 75 else "failed",
                "success_rate": success_rate,
                "total_questions": len(qa_results),
                "passed_questions": passed_count,
                "qa_results": qa_results
            }
            
            logger.info(f"✓ 質問応答フローテスト: {success_rate:.1f}% 成功率")
            
        except Exception as e:
            self.test_results["tests"][test_name] = {
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"質問応答フローテストでエラー: {e}")
    
    async def test_management_functions(self):
        """管理機能テスト"""
        logger.info("--- 管理機能テスト ---")
        test_name = "management_functions"
        
        try:
            # システム状態の取得
            status_response = requests.get(f"{self.api_base_url}/api/management/status")
            status_ok = status_response.status_code == 200
            
            # 文書一覧の取得
            docs_response = requests.get(f"{self.api_base_url}/api/management/documents")
            docs_ok = docs_response.status_code == 200
            
            # ログの取得
            logs_response = requests.get(f"{self.api_base_url}/api/management/logs?limit=10")
            logs_ok = logs_response.status_code == 200
            
            self.test_results["tests"][test_name] = {
                "status": "passed" if all([status_ok, docs_ok, logs_ok]) else "failed",
                "status_api": status_ok,
                "documents_api": docs_ok,
                "logs_api": logs_ok,
                "status_data": status_response.json() if status_ok else None,
                "documents_count": len(docs_response.json().get("documents", [])) if docs_ok else 0
            }
            
            logger.info(f"✓ 管理機能テスト: {'成功' if self.test_results['tests'][test_name]['status'] == 'passed' else '失敗'}")
            
        except Exception as e:
            self.test_results["tests"][test_name] = {
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"管理機能テストでエラー: {e}")
    
    async def test_error_handling(self):
        """エラーハンドリングテスト"""
        logger.info("--- エラーハンドリングテスト ---")
        test_name = "error_handling"
        
        try:
            error_tests = []
            
            # 1. 不正なリクエストのテスト
            invalid_request = requests.post(
                f"{self.api_base_url}/api/question",
                json={"invalid_field": "test"}
            )
            error_tests.append({
                "test": "invalid_request",
                "status_code": invalid_request.status_code,
                "handled_correctly": invalid_request.status_code == 400
            })
            
            # 2. 空の質問のテスト
            empty_question = requests.post(
                f"{self.api_base_url}/api/question",
                json={"question": ""}
            )
            error_tests.append({
                "test": "empty_question",
                "status_code": empty_question.status_code,
                "handled_correctly": empty_question.status_code in [400, 422]
            })
            
            # 3. 存在しないエンドポイントのテスト
            not_found = requests.get(f"{self.api_base_url}/api/nonexistent")
            error_tests.append({
                "test": "not_found_endpoint",
                "status_code": not_found.status_code,
                "handled_correctly": not_found.status_code == 404
            })
            
            # エラーハンドリングの成功率
            handled_correctly = sum(1 for t in error_tests if t["handled_correctly"])
            error_handling_rate = (handled_correctly / len(error_tests)) * 100
            
            self.test_results["tests"][test_name] = {
                "status": "passed" if error_handling_rate >= 80 else "failed",
                "error_handling_rate": error_handling_rate,
                "error_tests": error_tests
            }
            
            logger.info(f"✓ エラーハンドリングテスト: {error_handling_rate:.1f}% 正常処理")
            
        except Exception as e:
            self.test_results["tests"][test_name] = {
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"エラーハンドリングテストでエラー: {e}")
    
    async def test_performance(self):
        """パフォーマンステスト"""
        logger.info("--- パフォーマンステスト ---")
        test_name = "performance"
        
        try:
            # 複数の質問を並行実行
            test_question = "スパコンの使い方を教えてください"
            concurrent_requests = 3
            
            async def make_request():
                start_time = time.time()
                response = requests.post(
                    f"{self.api_base_url}/api/question",
                    json={"question": test_question},
                    timeout=30
                )
                end_time = time.time()
                return {
                    "response_time": end_time - start_time,
                    "status_code": response.status_code,
                    "success": response.status_code == 200
                }
            
            # 並行リクエストの実行
            tasks = [make_request() for _ in range(concurrent_requests)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 結果の分析
            successful_requests = [r for r in results if isinstance(r, dict) and r.get("success")]
            avg_response_time = sum(r["response_time"] for r in successful_requests) / len(successful_requests) if successful_requests else 0
            max_response_time = max(r["response_time"] for r in successful_requests) if successful_requests else 0
            
            # パフォーマンス基準の評価
            performance_ok = avg_response_time < 10.0 and max_response_time < 15.0
            
            self.test_results["tests"][test_name] = {
                "status": "passed" if performance_ok else "failed",
                "concurrent_requests": concurrent_requests,
                "successful_requests": len(successful_requests),
                "avg_response_time": avg_response_time,
                "max_response_time": max_response_time,
                "performance_criteria_met": performance_ok
            }
            
            logger.info(f"✓ パフォーマンステスト: 平均{avg_response_time:.2f}秒, 最大{max_response_time:.2f}秒")
            
        except Exception as e:
            self.test_results["tests"][test_name] = {
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"パフォーマンステストでエラー: {e}")
    
    async def test_offline_operation(self):
        """オフライン動作テスト"""
        logger.info("--- オフライン動作テスト ---")
        test_name = "offline_operation"
        
        try:
            # 既存のデータでオフライン動作を確認
            # （実際のネットワーク切断は行わず、ローカルデータでの動作を確認）
            
            # ローカルモデルの使用確認
            health_response = requests.get(f"{self.api_base_url}/health")
            health_data = health_response.json() if health_response.status_code == 200 else {}
            
            # 埋め込みモデルとベクトルDBの状態確認
            embedding_model_ok = health_data.get("components", {}).get("embedding_model") == "healthy"
            vector_db_ok = health_data.get("components", {}).get("database") == "healthy"
            
            # ローカルデータでの質問応答テスト
            offline_question = "テスト質問"
            offline_response = requests.post(
                f"{self.api_base_url}/api/question",
                json={"question": offline_question},
                timeout=10
            )
            offline_qa_ok = offline_response.status_code == 200
            
            offline_capability = all([embedding_model_ok, vector_db_ok, offline_qa_ok])
            
            self.test_results["tests"][test_name] = {
                "status": "passed" if offline_capability else "failed",
                "embedding_model_healthy": embedding_model_ok,
                "vector_db_healthy": vector_db_ok,
                "offline_qa_working": offline_qa_ok,
                "health_status": health_data
            }
            
            logger.info(f"✓ オフライン動作テスト: {'成功' if offline_capability else '失敗'}")
            
        except Exception as e:
            self.test_results["tests"][test_name] = {
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"オフライン動作テストでエラー: {e}")
    
    async def trigger_document_collection(self, url: str) -> Dict[str, Any]:
        """文書収集をトリガー"""
        try:
            response = requests.post(
                f"{self.api_base_url}/api/management/crawl",
                json={"url": url, "max_depth": 2},
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def validate_answer(self, answer_data: Dict[str, Any], test_case: Dict[str, Any]) -> Dict[str, Any]:
        """回答の検証"""
        validation = {
            "passed": False,
            "checks": {}
        }
        
        answer_text = answer_data.get("answer", "")
        sources = answer_data.get("sources", [])
        
        # 基本的な回答の存在確認
        validation["checks"]["has_answer"] = len(answer_text.strip()) > 0
        
        # ソース情報の確認（要件1.4）
        validation["checks"]["has_sources"] = len(sources) > 0
        
        # 期待されるキーワードの確認
        if test_case.get("expect_no_info"):
            # 情報が見つからない場合の適切な応答（要件1.3）
            validation["checks"]["appropriate_no_info_response"] = any(
                phrase in answer_text.lower() 
                for phrase in ["見つかりません", "情報がありません", "わかりません", "不明"]
            )
        else:
            # 期待されるキーワードが含まれているか
            expected_keywords = test_case.get("expected_keywords", [])
            if expected_keywords:
                keyword_found = any(
                    keyword.lower() in answer_text.lower() 
                    for keyword in expected_keywords
                )
                validation["checks"]["contains_expected_keywords"] = keyword_found
        
        # 処理時間の確認
        processing_time = answer_data.get("processing_time", 0)
        validation["checks"]["reasonable_processing_time"] = processing_time < 10.0
        
        # 全体的な検証結果
        validation["passed"] = all(validation["checks"].values())
        
        return validation
    
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
        results_file = "end_to_end_test_results.json"
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, ensure_ascii=False, indent=2)
            logger.info(f"テスト結果を保存しました: {results_file}")
        except Exception as e:
            logger.error(f"テスト結果の保存に失敗: {e}")

async def main():
    """メイン実行関数"""
    test_runner = EndToEndTest()
    results = await test_runner.run_all_tests()
    
    # 結果の表示
    print("\n" + "="*50)
    print("エンドツーエンドテスト完了")
    print("="*50)
    print(f"総合結果: {results['summary']['overall_status']}")
    print(f"成功率: {results['summary']['success_rate']:.1f}%")
    print(f"詳細結果: end_to_end_test_results.json")
    
    # 終了コード
    return 0 if results['summary']['overall_status'] == 'PASSED' else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)