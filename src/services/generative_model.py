"""
Generative AI model service for answer generation.

This module implements the generative AI model functionality using local
Transformers models. It provides answer generation based on search context
and supports Japanese language processing.
"""

import os
import time
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    GenerationConfig
)

from ..models.interfaces import GenerativeModelInterface
from ..models.core import Answer
from ..models.exceptions import GenerationError, ModelLoadError
from ..utils.config import get_config


logger = logging.getLogger(__name__)


class GenerativeModelService(GenerativeModelInterface):
    """
    Service for generating answers using local generative AI models.
    
    This service handles loading and using local generative models for
    answer generation based on retrieved context. It supports Japanese
    language models and provides configurable generation parameters.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the generative model service.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.model = None
        self.tokenizer = None
        self.generator = None
        self.model_name = None
        self.device = None
        self.generation_config = None
        
        # Load configuration
        self._load_config()
        
        # Initialize model if auto-load is enabled
        if self.config.get("generation", {}).get("auto_load", True):
            self.load_local_model(self.model_path)
    
    def _load_config(self) -> None:
        """Load configuration settings."""
        generation_config = self.config.get("generation", {})
        
        self.model_name = generation_config.get(
            "model_name", 
            "rinna/japanese-gpt2-medium"
        )
        self.model_path = generation_config.get(
            "model_path", 
            "./models/generation"
        )
        self.max_length = generation_config.get(
            "max_length", 
            512
        )
        self.temperature = generation_config.get(
            "temperature", 
            0.7
        )
        self.top_p = generation_config.get(
            "top_p", 
            0.9
        )
        self.device = generation_config.get(
            "device", 
            "cpu"
        )
        
        # Ensure device is available
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
    
    def load_local_model(self, model_path: str) -> bool:
        """
        Load a local generative model.
        
        Args:
            model_path: Path to the model files or model name
            
        Returns:
            True if model loaded successfully, False otherwise
            
        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            logger.info(f"Loading generative model from: {model_path}")
            start_time = time.time()
            
            # Determine if path is local directory or model name
            if os.path.exists(model_path):
                model_source = model_path
                logger.info(f"Loading from local path: {model_source}")
            else:
                model_source = self.model_name
                logger.info(f"Loading from model hub: {model_source}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                trust_remote_code=True,
                cache_dir=self.model_path if not os.path.exists(model_path) else None
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_source,
                trust_remote_code=True,
                torch_dtype=torch.float32,  # Use float32 for CPU compatibility
                device_map=None,  # Manual device placement
                cache_dir=self.model_path if not os.path.exists(model_path) else None
            )
            
            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Create generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float32
            )
            
            # Setup generation configuration
            self.generation_config = GenerationConfig(
                max_length=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                length_penalty=1.0
            )
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to load generative model: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e
    
    def generate_answer(self, query: str, context: List[str]) -> Answer:
        """
        Generate an answer based on query and context.
        
        Args:
            query: User question
            context: List of relevant context strings
            
        Returns:
            Generated answer with metadata
            
        Raises:
            GenerationError: If answer generation fails
        """
        if not self.model or not self.tokenizer:
            raise GenerationError("Model not loaded. Call load_local_model() first.")
        
        try:
            start_time = time.time()
            logger.info(f"Generating answer for query: {query[:100]}...")
            
            # Prepare prompt with context
            prompt = self._create_prompt(query, context)
            
            # Generate answer
            generated_text = self._generate_text(prompt)
            
            # Extract answer from generated text
            answer_text = self._extract_answer(generated_text, prompt)
            
            # Calculate confidence score (simplified heuristic)
            confidence = self._calculate_confidence(answer_text, context)
            
            # Extract source information
            sources = self._extract_sources(context)
            
            processing_time = time.time() - start_time
            
            answer = Answer(
                text=answer_text,
                sources=sources,
                confidence=confidence,
                processing_time=processing_time
            )
            
            logger.info(f"Answer generated successfully in {processing_time:.2f} seconds")
            return answer
            
        except Exception as e:
            error_msg = f"Failed to generate answer: {str(e)}"
            logger.error(error_msg)
            raise GenerationError(error_msg) from e
    
    def _create_prompt(self, query: str, context: List[str]) -> str:
        """
        Create a prompt for answer generation.
        
        Args:
            query: User question
            context: List of context strings
            
        Returns:
            Formatted prompt string
        """
        # Limit context to prevent token overflow
        max_context_length = 2000  # Approximate character limit
        combined_context = ""
        
        for ctx in context:
            if len(combined_context) + len(ctx) < max_context_length:
                combined_context += ctx + "\n\n"
            else:
                break
        
        # Create structured prompt for Japanese Q&A
        prompt = f"""以下の情報を参考にして、質問に対して正確で役立つ回答を日本語で生成してください。

参考情報:
{combined_context.strip()}

質問: {query}

回答: """
        
        return prompt
    
    def _generate_text(self, prompt: str) -> str:
        """
        Generate text using the loaded model.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            inputs = inputs.to(self.device)
            
            # Limit input length to prevent memory issues
            max_input_length = 1024
            if inputs.shape[1] > max_input_length:
                inputs = inputs[:, -max_input_length:]
                logger.warning(f"Input truncated to {max_input_length} tokens")
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    generation_config=self.generation_config,
                    max_new_tokens=256,  # Limit new tokens
                    num_return_sequences=1,
                    early_stopping=True
                )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}")
            raise
    
    def _extract_answer(self, generated_text: str, prompt: str) -> str:
        """
        Extract the answer portion from generated text.
        
        Args:
            generated_text: Full generated text
            prompt: Original prompt
            
        Returns:
            Extracted answer text
        """
        # Remove the prompt from generated text
        if generated_text.startswith(prompt):
            answer = generated_text[len(prompt):].strip()
        else:
            # Fallback: look for answer marker
            answer_marker = "回答: "
            if answer_marker in generated_text:
                answer = generated_text.split(answer_marker)[-1].strip()
            else:
                answer = generated_text.strip()
        
        # Clean up the answer
        answer = self._clean_answer(answer)
        
        # Ensure minimum answer length
        if len(answer) < 10:
            answer = "申し訳ございませんが、提供された情報では十分な回答を生成できませんでした。"
        
        return answer
    
    def _clean_answer(self, answer: str) -> str:
        """
        Clean and format the generated answer.
        
        Args:
            answer: Raw answer text
            
        Returns:
            Cleaned answer text
        """
        # Remove excessive whitespace
        answer = " ".join(answer.split())
        
        # Remove incomplete sentences at the end
        sentences = answer.split("。")
        if len(sentences) > 1 and sentences[-1].strip() and not sentences[-1].strip().endswith(("。", "！", "？")):
            answer = "。".join(sentences[:-1]) + "。"
        
        # Ensure proper ending
        if not answer.endswith(("。", "！", "？")):
            answer += "。"
        
        return answer
    
    def _calculate_confidence(self, answer: str, context: List[str]) -> float:
        """
        Calculate confidence score for the generated answer.
        
        Args:
            answer: Generated answer
            context: Context used for generation
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Simple heuristic-based confidence calculation
        confidence = 0.5  # Base confidence
        
        # Increase confidence if answer is substantial
        if len(answer) > 50:
            confidence += 0.2
        
        # Increase confidence if context is substantial
        if len(context) > 0 and sum(len(ctx) for ctx in context) > 200:
            confidence += 0.2
        
        # Decrease confidence for generic responses
        generic_phrases = [
            "申し訳ございません",
            "わかりません",
            "情報がありません",
            "確認してください"
        ]
        
        if any(phrase in answer for phrase in generic_phrases):
            confidence -= 0.3
        
        # Ensure confidence is within bounds
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    def _extract_sources(self, context: List[str]) -> List[str]:
        """
        Extract source information from context.
        
        Args:
            context: List of context strings
            
        Returns:
            List of source identifiers
        """
        sources = []
        
        for i, ctx in enumerate(context):
            # Create a simple source identifier
            source_id = f"文書{i+1}"
            
            # Try to extract URL or title if available
            # This is a simplified implementation
            if "http" in ctx:
                # Extract URL if present
                import re
                urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ctx)
                if urls:
                    source_id = urls[0]
            
            sources.append(source_id)
        
        return sources
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        if not self.model:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p
        }
    
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.model:
            del self.model
            self.model = None
        
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        if self.generator:
            del self.generator
            self.generator = None
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Generative model unloaded successfully")


def create_generative_model_service(config_path: Optional[str] = None) -> GenerativeModelService:
    """
    Factory function to create a GenerativeModelService instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured GenerativeModelService instance
    """
    return GenerativeModelService(config_path)


def get_generative_model_service(config_path: Optional[str] = None) -> GenerativeModelService:
    """
    Get a singleton GenerativeModelService instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        GenerativeModelService instance
    """
    # Simple singleton pattern for development
    if not hasattr(get_generative_model_service, '_instance'):
        get_generative_model_service._instance = create_generative_model_service(config_path)
    
    return get_generative_model_service._instance