"""
Inference module for op-schemes.

Contains model configurations and prompts for VLM/LLM inference.
"""

from .models_config import (
    get_vlm_config,
    get_llm_config,
    create_dspy_lm,
    create_openai_client,
    VLMProviderEnum,
    LLMProviderEnum,
    vlm_config,
    dots_ocr_config,
)

from .prompts import (
    # DSPy Extraction Prompts
    METADATA_EXTRACTION_PROMPT,
    get_dspy_table_markers_prompt,
    INBILL_EXTRACTION_INSTRUCTIONS,
    get_dspy_credit_note_instructions,
    get_dspy_nontabular_rewards_prompt,
    get_dspy_product_matching_prompt,
    # Image Company Recognition Prompts
    COMPANY_RECOGNITION_DEFAULT_SYSTEM_PROMPT,
    COMPANY_IDENTIFICATION_PROMPT,
    COMPANY_RECOGNITION_SHORT_USER_PROMPT,
    # Product Mapping Prompts
    get_product_mapping_prompt,
    PRODUCT_MAPPING_LLM_SYSTEM_PROMPT,
)

__all__ = [
    # models_config exports
    "get_vlm_config",
    "get_llm_config",
    "create_dspy_lm",
    "create_openai_client",
    "VLMProviderEnum",
    "LLMProviderEnum",
    "vlm_config",
    "dots_ocr_config",
    # prompts exports - DSPy Extraction
    "METADATA_EXTRACTION_PROMPT",
    "get_dspy_table_markers_prompt",
    "INBILL_EXTRACTION_INSTRUCTIONS",
    "get_dspy_credit_note_instructions",
    "get_dspy_nontabular_rewards_prompt",
    "get_dspy_product_matching_prompt",
    # prompts exports - Image Company Recognition
    "COMPANY_RECOGNITION_DEFAULT_SYSTEM_PROMPT",
    "COMPANY_IDENTIFICATION_PROMPT",
    "COMPANY_RECOGNITION_SHORT_USER_PROMPT",
    # prompts exports - Product Mapping
    "get_product_mapping_prompt",
    "PRODUCT_MAPPING_LLM_SYSTEM_PROMPT",
]
