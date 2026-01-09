"""
Centralized model configuration and initialization.

This module provides configuration for VLM and LLM models.
It eliminates the need to repeatedly define model endpoints and configurations across features.

Usage Examples:
    # For VLM-based features:
    from inference import get_vlm_config, VLMProviderEnum
    config = get_vlm_config(VLMProviderEnum.qwen3_vl)

    # For DSPy-based features (factory functions):
    from inference import create_dspy_lm

    lm = create_dspy_lm(api_base="http://localhost:1234/v1", model_name="qwen3-4b")
"""

import dspy
from typing import Dict, Optional, Literal
from enum import Enum
from logger import logger


# ============================================================================
# VLM Configuration (Vision Language Model)
# ============================================================================

class VLMConfig:
    """Configuration for VLM endpoints and models"""
    
    # VLM Server Configuration
    URL = "http://13.127.71.115:8080/v1/chat/completions"
    MODEL = "longlichi"
    
    # Default VLM Parameters
    TEMPERATURE = 0.1
    MAX_TOKENS = 2000
    TIMEOUT = 60  # seconds


# Expose VLM config as a dict for easy access
vlm_config = {
    "url": VLMConfig.URL,
    "model": VLMConfig.MODEL,
    "temperature": VLMConfig.TEMPERATURE,
    "max_tokens": VLMConfig.MAX_TOKENS,
    "timeout": VLMConfig.TIMEOUT,
}


# ============================================================================
# DotsOCR Configuration (OCR Model)
# ============================================================================

class DotsOCRConfig:
    """Configuration for DotsOCR endpoints and models"""
    
    # DotsOCR Server Configuration
    URL = "http://13.200.146.36:8000/v1"
    # The DotsOCR server uses "model" as the model name
    MODEL = "model"
    
    # Default DotsOCR Parameters
    TEMPERATURE = 0.1
    TOP_P = 0.9
    MAX_COMPLETION_TOKENS = 32768
    TIMEOUT = 120  # seconds (OCR can take longer)


# Expose DotsOCR config as a dict for easy access
dots_ocr_config = {
    "url": DotsOCRConfig.URL,
    "model": DotsOCRConfig.MODEL,
    "temperature": DotsOCRConfig.TEMPERATURE,
    "top_p": DotsOCRConfig.TOP_P,
    "max_completion_tokens": DotsOCRConfig.MAX_COMPLETION_TOKENS,
    "timeout": DotsOCRConfig.TIMEOUT,
}


# ============================================================================
# LLM Configuration (for DSPy and OpenAI-compatible models)
# ============================================================================

class LLMProviderEnum(str, Enum):
    """Enum for different LLM providers"""
    ollama = "ollama"
    lm_studio = "lm_studio"
    aws_hosted = "aws_hosted"
    runpod = "runpod"


class VLMProviderEnum(str, Enum):
    """Enum for different VLM providers"""
    qwen3_vl = "qwen3_vl"


class Qwen3VLConfig:
    """Configuration for Qwen3-VL model (Vision Language Model)"""

    # VLM Server Configuration
    URL = "https://cehuu60kzh9mic-8000.proxy.runpod.net/v1/chat/completions"
    MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"

    # VLM Parameters
    TEMPERATURE = 0.1
    MAX_TOKENS = 2048
    REQUEST_TIMEOUT = 120  # seconds
    MAX_RETRIES = 3

    # Image Processing Configuration
    ENABLE_PREPROCESSING = True
    CONVERT_TO_GRAYSCALE = False
    ENHANCE_CONTRAST = False
    CONTRAST_FACTOR = 2.0
    MAX_IMAGE_DIMENSION = 2048  # pixels
    MAX_ENCODED_SIZE_MB = 10  # megabytes
    JPEG_QUALITY = 85  # 0-100


class VLMConfig:
    """Configuration for different VLM providers"""

    PROVIDERS = {
        VLMProviderEnum.qwen3_vl: {
            "url": Qwen3VLConfig.URL,
            "model": Qwen3VLConfig.MODEL,
            "temperature": Qwen3VLConfig.TEMPERATURE,
            "max_tokens": Qwen3VLConfig.MAX_TOKENS,
            "request_timeout": Qwen3VLConfig.REQUEST_TIMEOUT,
            "max_retries": Qwen3VLConfig.MAX_RETRIES,
            # Image processing config
            "enable_preprocessing": Qwen3VLConfig.ENABLE_PREPROCESSING,
            "convert_to_grayscale": Qwen3VLConfig.CONVERT_TO_GRAYSCALE,
            "enhance_contrast": Qwen3VLConfig.ENHANCE_CONTRAST,
            "contrast_factor": Qwen3VLConfig.CONTRAST_FACTOR,
            "max_image_dimension": Qwen3VLConfig.MAX_IMAGE_DIMENSION,
            "max_encoded_size_mb": Qwen3VLConfig.MAX_ENCODED_SIZE_MB,
            "jpeg_quality": Qwen3VLConfig.JPEG_QUALITY,
        }
    }


def get_vlm_config(provider: VLMProviderEnum = VLMProviderEnum.qwen3_vl) -> Dict:
    """
    Get VLM configuration for a specific provider.

    Args:
        provider: VLM provider enum (default: qwen3_vl)

    Returns:
        Dictionary with all VLM configuration keys
    """
    if provider not in VLMConfig.PROVIDERS:
        raise ValueError(f"Unknown VLM provider: {provider}")

    return VLMConfig.PROVIDERS[provider].copy()


class LLMConfig:
    """Configuration for different LLM providers"""
    
    PROVIDERS = {
        LLMProviderEnum.ollama: {
            "url": "http://localhost:11434/v1",
            "model": "qwen2.5:7b"
        },
        LLMProviderEnum.lm_studio: {
            "url": "http://192.168.1.16:1234/v1",
            "model": "ibm/granite-3.2-8b"
        },
        LLMProviderEnum.aws_hosted: {
            "url": "http://13.127.71.115:4000/v1",
            "model": "aws_hosted"
        },
        LLMProviderEnum.runpod: {
            "url": "https://sdkdp94ldzgcw2-8000.proxy.runpod.net/v1",
            "model": "Qwen/Qwen2.5-32B-Instruct"
        }
    }
    
    # Default LLM Parameters
    DEFAULT_TEMPERATURE = 0.1
    DEFAULT_MAX_TOKENS = 4096
    DEFAULT_SEED = 42


def get_llm_config(provider: LLMProviderEnum) -> Dict:
    """
    Get LLM configuration for a specific provider.
    
    Args:
        provider: LLM provider enum
        
    Returns:
        Dictionary with 'url' and 'model' keys
    """
    if provider not in LLMConfig.PROVIDERS:
        raise ValueError(f"Unknown LLM provider: {provider}")
    
    return LLMConfig.PROVIDERS[provider].copy()


def create_dspy_lm(
    api_base: str,
    model_name: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None
) -> dspy.LM:
    """
    Factory function to create a DSPy LM instance with OpenAI-compatible endpoint.
    
    Args:
        api_base: Base URL of the LLM API endpoint
        model_name: Name of the model to use
        temperature: Sampling temperature (default: 0.0 for deterministic)
        max_tokens: Maximum tokens in response (default: 4096)
        seed: Random seed for reproducibility (default: 42)
        
    Returns:
        Configured dspy.LM instance
        
    Example:
        lm = create_dspy_lm(
            api_base="http://localhost:1234/v1",
            model_name="qwen3-4b"
        )
    """
    temperature = temperature if temperature is not None else LLMConfig.DEFAULT_TEMPERATURE
    max_tokens = max_tokens if max_tokens is not None else LLMConfig.DEFAULT_MAX_TOKENS
    seed = seed if seed is not None else LLMConfig.DEFAULT_SEED
    
    logger.info(f"Creating DSPy LM: {api_base}, model={model_name}, temp={temperature}")
    
    lm = dspy.LM(
        model=f"openai/{model_name}",
        api_base=api_base,
        api_key="dummy",  # OpenAI-compatible servers often don't require real API key
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
        cache=False,  # Disable caching
    )
    
    return lm


def create_openai_client(api_base: str, api_key: str = "not-needed"):
    """
    Factory function to create an OpenAI client for non-DSPy use cases.
    
    Args:
        api_base: Base URL of the OpenAI-compatible API
        api_key: API key (default: "not-needed" for local servers)
        
    Returns:
        OpenAI client instance
        
    Example:
        from openai import OpenAI
        client = create_openai_client("http://localhost:1234/v1")
    """
    from openai import OpenAI
    
    logger.info(f"Creating OpenAI client: {api_base}")
    
    client = OpenAI(
        base_url=api_base,
        api_key=api_key
    )
    
    return client


# ============================================================================
# Utility Functions
# ============================================================================

def get_model_summary() -> Dict:
    """
    Get a summary of all configured models and endpoints.
    Useful for debugging and logging.
    
    Returns:
        Dictionary with VLM, DotsOCR and LLM configurations
    """
    return {
        "vlm": vlm_config,
        "dots_ocr": dots_ocr_config,
        "llm_providers": LLMConfig.PROVIDERS.copy(),
        "llm_defaults": {
            "temperature": LLMConfig.DEFAULT_TEMPERATURE,
            "max_tokens": LLMConfig.DEFAULT_MAX_TOKENS,
            "seed": LLMConfig.DEFAULT_SEED,
        }
    }


def log_model_config():
    """Log current model configuration for debugging"""
    summary = get_model_summary()
    logger.info("=" * 80)
    logger.info("MODEL CONFIGURATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"VLM Endpoint: {summary['vlm']['url']}")
    logger.info(f"VLM Model: {summary['vlm']['model']}")
    logger.info("-" * 80)
    logger.info(f"DotsOCR Endpoint: {summary['dots_ocr']['url']}")
    logger.info(f"DotsOCR Model: {summary['dots_ocr']['model']}")
    logger.info("-" * 80)
    for provider, config in summary['llm_providers'].items():
        logger.info(f"LLM Provider '{provider}': {config['url']} ({config['model']})")
    logger.info("=" * 80)


# ============================================================================
# Initialization
# ============================================================================

# Log configuration on module import
log_model_config()
