"""
Tests for the generative model service.

This module contains unit tests for the GenerativeModelService class,
testing model loading, answer generation, and error handling.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from src.services.generative_model import GenerativeModelService
from src.models.core import Answer
from src.models.exceptions import GenerationError, ModelLoadError


class TestGenerativeModelService:
    """Test cases for GenerativeModelService."""
    
    @pytest.fixture
    def generative_service(self):
        """Create a GenerativeModelService instance."""
        with patch('src.services.generative_model.get_config') as mock_get_config:
            mock_get_config.return_value = {
                "generation": {
                    "model_name": "rinna/japanese-gpt2-medium",
                    "model_path": "./models/generation",
                    "max_length": 512,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "device": "cpu",
                    "auto_load": False
                }
            }
            return GenerativeModelService()
    
    def test_initialization(self, generative_service):
        """Test service initialization."""
        assert generative_service.model is None
        assert generative_service.tokenizer is None
        assert generative_service.generator is None
        assert generative_service.model_name == "rinna/japanese-gpt2-medium"
        assert generative_service.device == "cpu"
    
    def test_config_loading(self, generative_service):
        """Test configuration loading."""
        assert generative_service.max_length == 512
        assert generative_service.temperature == 0.7
        assert generative_service.top_p == 0.9
        assert generative_service.device == "cpu"
    
    @patch('src.services.generative_model.torch')
    @patch('src.services.generative_model.AutoTokenizer')
    @patch('src.services.generative_model.AutoModelForCausalLM')
    @patch('src.services.generative_model.pipeline')
    def test_load_local_model_success(self, mock_pipeline, mock_model_class, 
                                    mock_tokenizer_class, mock_torch, generative_service):
        """Test successful model loading."""
        # Setup mocks
        mock_torch.cuda.is_available.return_value = False
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "[EOS]"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_generator = Mock()
        mock_pipeline.return_value = mock_generator
        
        # Test model loading
        result = generative_service.load_local_model("./test/model")
        
        assert result is True
        assert generative_service.tokenizer is not None
        assert generative_service.model is not None
        assert generative_service.generator is not None
        
        # Verify tokenizer setup
        assert mock_tokenizer.pad_token == "[EOS]"
        
        # Verify model calls
        mock_tokenizer_class.from_pretrained.assert_called_once()
        mock_model_class.from_pretrained.assert_called_once()
        mock_pipeline.assert_called_once()
    
    @patch('src.services.generative_model.AutoTokenizer')
    def test_load_local_model_failure(self, mock_tokenizer_class, generative_service):
        """Test model loading failure."""
        # Setup mock to raise exception
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Model not found")
        
        # Test model loading failure
        with pytest.raises(ModelLoadError):
            generative_service.load_local_model("./nonexistent/model")
    
    def test_generate_answer_without_model(self, generative_service):
        """Test answer generation without loaded model."""
        with pytest.raises(GenerationError, match="Model not loaded"):
            generative_service.generate_answer("テスト質問", ["コンテキスト"])
    
    @patch('src.services.generative_model.torch')
    def test_generate_answer_success(self, mock_torch, generative_service):
        """Test successful answer generation."""
        # Setup mock model and tokenizer
        mock_tokenizer = Mock()
        
        # Create a proper mock tensor with shape attribute
        mock_input_tensor = Mock()
        mock_input_tensor.shape = [1, 10]  # batch_size=1, seq_len=10
        mock_input_tensor.to.return_value = mock_input_tensor
        mock_tokenizer.encode.return_value = mock_input_tensor
        
        mock_tokenizer.decode.return_value = "以下の情報を参考にして、質問に対して正確で役立つ回答を日本語で生成してください。\n\n参考情報:\nテストコンテキスト\n\n質問: テスト質問\n\n回答: これはテスト回答です。"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        
        mock_model = Mock()
        mock_outputs = Mock()
        mock_model.generate.return_value = [mock_outputs]
        
        generative_service.model = mock_model
        generative_service.tokenizer = mock_tokenizer
        generative_service.device = "cpu"
        
        # Setup torch mocks
        mock_torch.no_grad.return_value.__enter__ = Mock()
        mock_torch.no_grad.return_value.__exit__ = Mock()
        
        # Test answer generation
        answer = generative_service.generate_answer("テスト質問", ["テストコンテキスト"])
        
        assert isinstance(answer, Answer)
        assert answer.text == "これはテスト回答です。"
        assert len(answer.sources) > 0
        assert 0.0 <= answer.confidence <= 1.0
        assert answer.processing_time > 0
    
    def test_create_prompt(self, generative_service):
        """Test prompt creation."""
        query = "スパコンの使い方は？"
        context = ["スパコンは高性能計算機です。", "利用にはアカウントが必要です。"]
        
        prompt = generative_service._create_prompt(query, context)
        
        assert "以下の情報を参考にして" in prompt
        assert query in prompt
        assert context[0] in prompt
        assert context[1] in prompt
        assert "回答: " in prompt
    
    def test_extract_answer(self, generative_service):
        """Test answer extraction from generated text."""
        prompt = "質問: テスト\n回答: "
        generated_text = prompt + "これはテスト回答です。"
        
        answer = generative_service._extract_answer(generated_text, prompt)
        
        assert answer == "これはテスト回答です。"
    
    def test_clean_answer(self, generative_service):
        """Test answer cleaning."""
        # Test with incomplete sentence
        raw_answer = "これはテスト回答です。追加の情報"
        cleaned = generative_service._clean_answer(raw_answer)
        assert cleaned == "これはテスト回答です。"
        
        # Test with proper ending
        raw_answer = "これは完全な回答です。"
        cleaned = generative_service._clean_answer(raw_answer)
        assert cleaned == "これは完全な回答です。"
        
        # Test without ending punctuation
        raw_answer = "これは回答です"
        cleaned = generative_service._clean_answer(raw_answer)
        assert cleaned == "これは回答です。"
    
    def test_calculate_confidence(self, generative_service):
        """Test confidence calculation."""
        # Test with substantial answer and context
        answer = "これは詳細で有用な回答です。多くの情報を含んでいます。"
        context = ["長いコンテキスト情報がここにあります。" * 10]
        confidence = generative_service._calculate_confidence(answer, context)
        assert 0.5 <= confidence <= 1.0
        
        # Test with generic response
        answer = "申し訳ございませんが、わかりません。"
        context = ["短いコンテキスト"]
        confidence = generative_service._calculate_confidence(answer, context)
        assert confidence < 0.5
    
    def test_extract_sources(self, generative_service):
        """Test source extraction."""
        context = [
            "これは最初の文書です。",
            "これは2番目の文書で、http://example.com のURLを含みます。",
            "これは3番目の文書です。"
        ]
        
        sources = generative_service._extract_sources(context)
        
        assert len(sources) == 3
        assert sources[0] == "文書1"
        assert "http://example.com" in sources[1]
        assert sources[2] == "文書3"
    
    def test_get_model_info_not_loaded(self, generative_service):
        """Test model info when model is not loaded."""
        info = generative_service.get_model_info()
        assert info["status"] == "not_loaded"
    
    def test_get_model_info_loaded(self, generative_service):
        """Test model info when model is loaded."""
        # Mock loaded model
        generative_service.model = Mock()
        
        info = generative_service.get_model_info()
        
        assert info["status"] == "loaded"
        assert info["model_name"] == "rinna/japanese-gpt2-medium"
        assert info["device"] == "cpu"
    
    def test_unload_model(self, generative_service):
        """Test model unloading."""
        # Setup mock loaded components
        generative_service.model = Mock()
        generative_service.tokenizer = Mock()
        generative_service.generator = Mock()
        
        generative_service.unload_model()
        
        assert generative_service.model is None
        assert generative_service.tokenizer is None
        assert generative_service.generator is None


class TestGenerativeModelIntegration:
    """Integration tests for GenerativeModelService."""
    
    def test_prompt_creation_with_long_context(self):
        """Test prompt creation with very long context."""
        with patch('src.services.generative_model.get_config') as mock_get_config:
            mock_get_config.return_value = {
                "generation": {
                    "model_name": "rinna/japanese-gpt2-medium",
                    "model_path": "./test_models/generation",
                    "max_length": 256,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "device": "cpu",
                    "auto_load": False
                }
            }
            service = GenerativeModelService()
        
        query = "テスト質問"
        long_context = ["非常に長いコンテキスト文書です。" * 100] * 10
        
        prompt = service._create_prompt(query, long_context)
        
        # Verify prompt is not excessively long
        assert len(prompt) < 3000  # Should be truncated
        assert query in prompt
    
    def test_answer_extraction_edge_cases(self):
        """Test answer extraction with edge cases."""
        with patch('src.services.generative_model.get_config') as mock_get_config:
            mock_get_config.return_value = {
                "generation": {
                    "model_name": "rinna/japanese-gpt2-medium",
                    "model_path": "./test_models/generation",
                    "max_length": 256,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "device": "cpu",
                    "auto_load": False
                }
            }
            service = GenerativeModelService()
        
        # Test with no clear answer marker
        generated_text = "これは生成されたテキストです。回答が含まれています。"
        prompt = "質問: テスト"
        
        answer = service._extract_answer(generated_text, prompt)
        assert len(answer) >= 10  # Should have minimum length
        
        # Test with very short generated text
        generated_text = "短い"
        answer = service._extract_answer(generated_text, prompt)
        assert "申し訳ございません" in answer  # Should use fallback