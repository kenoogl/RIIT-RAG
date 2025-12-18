"""
Integration tests for the generative model service.

This module contains integration tests that verify the GenerativeModelService
works correctly with real configuration and mock models.
"""

import pytest
import tempfile
import os
from unittest.mock import patch, Mock

from src.services.generative_model import GenerativeModelService
from src.models.core import Answer


class TestGenerativeModelIntegration:
    """Integration tests for GenerativeModelService."""
    
    def test_service_creation_with_real_config(self):
        """Test service creation with real configuration file."""
        # Test with default configuration (since config caching makes custom config testing complex)
        service = GenerativeModelService()
        
        # Verify that service loads with expected defaults from config.yaml
        assert service.model_name == "rinna/japanese-gpt2-medium"
        assert service.device == "cpu"
        assert service.max_length == 512  # This is the actual default in config.yaml
        assert service.temperature == 0.7
        assert service.top_p == 0.9
    
    def test_end_to_end_answer_generation_flow(self):
        """Test the complete answer generation flow with mocked components."""
        with patch('src.services.generative_model.get_config') as mock_get_config:
            mock_get_config.return_value = {
                "generation": {
                    "model_name": "test-model",
                    "model_path": "./test_models",
                    "max_length": 256,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "device": "cpu",
                    "auto_load": False
                }
            }
            
            service = GenerativeModelService()
            
            # Mock the model loading and generation process
            with patch.object(service, 'load_local_model') as mock_load:
                with patch.object(service, '_generate_text') as mock_generate:
                    mock_load.return_value = True
                    mock_generate.return_value = "質問: スパコンの使い方は？\n\n回答: スパコンを使用するには、まずアカウントを作成する必要があります。"
                    
                    # Setup mock model components
                    service.model = Mock()
                    service.tokenizer = Mock()
                    
                    # Load model
                    result = service.load_local_model("./test_model")
                    assert result is True
                    
                    # Generate answer
                    query = "スパコンの使い方は？"
                    context = [
                        "スパコンは高性能計算機です。",
                        "利用にはアカウントが必要です。",
                        "バッチジョブとして計算を実行します。"
                    ]
                    
                    answer = service.generate_answer(query, context)
                    
                    assert isinstance(answer, Answer)
                    assert len(answer.text) > 0
                    assert len(answer.sources) == len(context)
                    assert 0.0 <= answer.confidence <= 1.0
                    assert answer.processing_time > 0
    
    def test_prompt_creation_with_japanese_content(self):
        """Test prompt creation with Japanese content."""
        with patch('src.services.generative_model.get_config') as mock_get_config:
            mock_get_config.return_value = {
                "generation": {
                    "auto_load": False
                }
            }
            
            service = GenerativeModelService()
            
            query = "九州大学のスパコンの利用方法を教えてください"
            context = [
                "九州大学情報基盤研究開発センターでは、スーパーコンピュータシステムを提供しています。",
                "利用するためには、まずアカウントの申請が必要です。",
                "計算ジョブはバッチシステムを通じて実行されます。"
            ]
            
            prompt = service._create_prompt(query, context)
            
            # Verify prompt structure
            assert "以下の情報を参考にして" in prompt
            assert query in prompt
            assert "回答: " in prompt
            
            # Verify all context is included
            for ctx in context:
                assert ctx in prompt
    
    def test_answer_extraction_with_japanese_text(self):
        """Test answer extraction with Japanese generated text."""
        with patch('src.services.generative_model.get_config') as mock_get_config:
            mock_get_config.return_value = {
                "generation": {
                    "auto_load": False
                }
            }
            
            service = GenerativeModelService()
            
            prompt = "質問: テスト質問\n\n回答: "
            generated_text = prompt + "九州大学のスーパーコンピュータを利用するには、まずアカウント申請を行ってください。詳細は公式サイトをご確認ください。"
            
            answer = service._extract_answer(generated_text, prompt)
            
            assert "九州大学のスーパーコンピュータ" in answer
            assert answer.endswith("。")
            assert len(answer) > 10
    
    def test_confidence_calculation_with_various_scenarios(self):
        """Test confidence calculation with various answer scenarios."""
        with patch('src.services.generative_model.get_config') as mock_get_config:
            mock_get_config.return_value = {
                "generation": {
                    "auto_load": False
                }
            }
            
            service = GenerativeModelService()
            
            # Test high confidence scenario
            good_answer = "九州大学のスーパーコンピュータを利用するには、まずアカウント申請を行い、承認後にSSHでログインしてバッチジョブを投入します。"
            rich_context = [
                "九州大学情報基盤研究開発センターでは、研究者向けにスーパーコンピュータシステムを提供しています。" * 3,
                "利用申請は公式サイトから行うことができ、審査後にアカウントが発行されます。" * 3,
                "計算ジョブはSLURMバッチシステムを使用して実行されます。" * 3
            ]
            
            high_confidence = service._calculate_confidence(good_answer, rich_context)
            
            # Test low confidence scenario
            poor_answer = "申し訳ございませんが、詳細な情報がありません。"
            poor_context = ["短い情報"]
            
            low_confidence = service._calculate_confidence(poor_answer, poor_context)
            
            assert high_confidence > low_confidence
            assert 0.0 <= high_confidence <= 1.0
            assert 0.0 <= low_confidence <= 1.0
    
    def test_source_extraction_with_urls(self):
        """Test source extraction with URLs in context."""
        with patch('src.services.generative_model.get_config') as mock_get_config:
            mock_get_config.return_value = {
                "generation": {
                    "auto_load": False
                }
            }
            
            service = GenerativeModelService()
            
            context = [
                "九州大学の公式サイト https://www.kyushu-u.ac.jp/ で詳細を確認できます。",
                "スパコンの利用方法については https://www.cc.kyushu-u.ac.jp/scp/ をご覧ください。",
                "一般的な情報です。URLは含まれていません。"
            ]
            
            sources = service._extract_sources(context)
            
            assert len(sources) == 3
            assert "https://www.kyushu-u.ac.jp/" in sources[0]
            assert "https://www.cc.kyushu-u.ac.jp/scp/" in sources[1]
            assert sources[2] == "文書3"
    
    def test_model_info_functionality(self):
        """Test model information retrieval."""
        with patch('src.services.generative_model.get_config') as mock_get_config:
            mock_get_config.return_value = {
                "generation": {
                    "model_name": "test-model",
                    "model_path": "./test_models",
                    "max_length": 256,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "device": "cpu",
                    "auto_load": False
                }
            }
            
            service = GenerativeModelService()
            
            # Test when model is not loaded
            info = service.get_model_info()
            assert info["status"] == "not_loaded"
            
            # Test when model is loaded
            service.model = Mock()
            info = service.get_model_info()
            
            assert info["status"] == "loaded"
            assert info["model_name"] == "test-model"
            assert info["device"] == "cpu"
            assert info["max_length"] == 256
            assert info["temperature"] == 0.7
            assert info["top_p"] == 0.9
    
    def test_model_unloading(self):
        """Test model unloading functionality."""
        with patch('src.services.generative_model.get_config') as mock_get_config:
            mock_get_config.return_value = {
                "generation": {
                    "auto_load": False
                }
            }
            
            service = GenerativeModelService()
            
            # Setup mock loaded components
            service.model = Mock()
            service.tokenizer = Mock()
            service.generator = Mock()
            
            # Verify components are loaded
            assert service.model is not None
            assert service.tokenizer is not None
            assert service.generator is not None
            
            # Unload model
            with patch('src.services.generative_model.torch') as mock_torch:
                mock_torch.cuda.is_available.return_value = False
                service.unload_model()
            
            # Verify components are unloaded
            assert service.model is None
            assert service.tokenizer is None
            assert service.generator is None