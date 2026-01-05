"""
Inference API Server.

General-purpose inference server that hides prompts and model details
from clients. Clients only send base64 images and receive extracted text.

The server runs on port 8001.
"""

import base64
import tempfile
import os
from io import BytesIO
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from openai import OpenAI

from logger import logger
from inference.models_config import dots_ocr_config
from inference.prompts import DOTSOCR_PROMPT_MODES
from inference import get_vlm_config, VLMProviderEnum, get_llm_config, LLMProviderEnum
from inference.prompts import (
    COMPANY_IDENTIFICATION_PROMPT,
    COMPANY_RECOGNITION_SHORT_USER_PROMPT,
    PRODUCT_MAPPING_LLM_SYSTEM_PROMPT,
    get_product_mapping_prompt,
)
import requests
import json


# =============================================================================
# Request/Response Models
# =============================================================================

class OCRRequest(BaseModel):
    """Request model for OCR inference"""
    image_base64: str
    prompt_mode: Optional[Literal[
        "prompt_ocr",
        "prompt_layout_all_en",
        "prompt_layout_only_en",
        "prompt_grounding_ocr"
    ]] = "prompt_ocr"


class OCRResponse(BaseModel):
    """Response model for OCR inference"""
    text: str
    prompt_mode: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str


class ProductExtractionRequest(BaseModel):
    """Request model for product extraction"""
    image_base64: str
    markdown: str


class ProductExtractionResponse(BaseModel):
    """Response model for product extraction"""
    product_offers: list


class CompanyDetectionRequest(BaseModel):
    """Request model for company detection"""
    image_base64: str


class CompanyDetectionResponse(BaseModel):
    """Response model for company detection"""
    company: str


class ProductMappingRequest(BaseModel):
    """Request model for product mapping"""
    product_name: str
    category_pairs: list  # List of [category, type] pairs


class ProductMappingResponse(BaseModel):
    """Response model for product mapping"""
    matched_pairs: list  # List of {category, type} dicts


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Inference API",
    description="Internal inference service with hidden prompts",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Configuration
# =============================================================================

# OCR server configuration
OCR_URL = dots_ocr_config["url"]
OCR_MODEL = dots_ocr_config["model"]
OCR_TEMPERATURE = dots_ocr_config["temperature"]
OCR_TOP_P = dots_ocr_config["top_p"]
OCR_MAX_TOKENS = dots_ocr_config["max_completion_tokens"]
OCR_TIMEOUT = dots_ocr_config["timeout"]

# VLM server configuration for company detection
_vlm_config = get_vlm_config(VLMProviderEnum.qwen3_vl)
VLM_URL = _vlm_config["url"]
VLM_MODEL = _vlm_config["model"]
VLM_TEMPERATURE = _vlm_config["temperature"]
VLM_MAX_TOKENS = _vlm_config["max_tokens"]

# LLM server configuration for product mapping
_llm_config = get_llm_config(LLMProviderEnum.runpod)
LLM_URL = _llm_config["url"]
LLM_MODEL = _llm_config["model"]


# =============================================================================
# Helper Functions
# =============================================================================

def base64_to_image(image_base64: str) -> Image.Image:
    """
    Convert base64 string to PIL Image.

    Args:
        image_base64: Base64 encoded image string

    Returns:
        PIL Image object
    """
    # Handle data URL prefix if present
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]

    image_data = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_data))
    return image


def image_to_base64_url(image: Image.Image) -> str:
    """
    Convert PIL Image to base64 URL format for OpenAI API.

    Args:
        image: PIL Image object

    Returns:
        Base64 encoded image URL string
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"


def image_to_jpeg_base64(image: Image.Image) -> str:
    """
    Convert PIL Image to JPEG base64 string for VLM.

    Args:
        image: PIL Image object

    Returns:
        Base64 encoded JPEG string
    """
    # Resize to max 1024x1024 while maintaining aspect ratio
    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

    # Convert RGBA to RGB if necessary
    if image.mode == 'RGBA':
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    # Save to bytes buffer
    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=85)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded


def run_ocr_inference(image: Image.Image, prompt: str) -> str:
    """
    Run OCR inference on backend server.

    Args:
        image: PIL Image object
        prompt: Text prompt for OCR extraction

    Returns:
        Extracted text from the image

    Raises:
        HTTPException: If inference fails
    """
    try:
        client = OpenAI(api_key="0", base_url=OCR_URL)

        # Convert image to base64 URL
        image_url = image_to_base64_url(image)

        # Build messages in OpenAI chat format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                    {"type": "text", "text": f"<|img|><|imgpad|><|endofimg|>{prompt}"}
                ],
            }
        ]

        # Try to get the correct model name
        try:
            models_response = client.models.list()
            available_models = [model.id for model in models_response.data]
            logger.info(f"Available models: {available_models}")
            model_to_use = available_models[0] if available_models else OCR_MODEL
            logger.info(f"Using model: {model_to_use}")
        except Exception as model_list_error:
            logger.warning(f"Could not list models: {model_list_error}. Using configured model: {OCR_MODEL}")
            model_to_use = OCR_MODEL

        # Make request to server
        response = client.chat.completions.create(
            messages=messages,
            model=model_to_use,
            max_completion_tokens=OCR_MAX_TOKENS,
            temperature=OCR_TEMPERATURE,
            top_p=OCR_TOP_P
        )

        response_text = response.choices[0].message.content
        logger.info(f"OCR inference successful, response length: {len(response_text)}")
        return response_text

    except Exception as e:
        logger.error(f"OCR inference failed: {e}")
        # Return generic error message - don't leak internal details
        raise HTTPException(status_code=500, detail="Request timed out")


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="healthy", service="inference")


@app.post("/ocr", response_model=OCRResponse)
async def ocr_inference(request: OCRRequest):
    """
    Run OCR inference on a base64-encoded image.

    The prompt is determined server-side based on prompt_mode,
    keeping prompts hidden from clients.

    Args:
        request: OCRRequest with image_base64 and optional prompt_mode

    Returns:
        OCRResponse with extracted text
    """
    # Validate prompt_mode
    if request.prompt_mode not in DOTSOCR_PROMPT_MODES:
        raise HTTPException(
            status_code=400,
            detail="Invalid request"
        )

    # Get the prompt (hidden from client)
    prompt = DOTSOCR_PROMPT_MODES[request.prompt_mode]
    logger.info(f"Using prompt_mode: {request.prompt_mode}")

    try:
        # Convert base64 to PIL Image
        image = base64_to_image(request.image_base64)
        logger.info(f"Image converted successfully: {image.size}, mode: {image.mode}")

        # Run inference
        result_text = run_ocr_inference(image, prompt)

        return OCRResponse(text=result_text, prompt_mode=request.prompt_mode)

    except base64.binascii.Error as e:
        logger.error(f"Invalid base64 encoding: {e}")
        raise HTTPException(status_code=400, detail="Invalid image data")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Text extraction failed")


@app.post("/extract-products", response_model=ProductExtractionResponse)
async def extract_products(request: ProductExtractionRequest):
    """
    Run product extraction using the internal VLM+LLM pipeline.

    The actual extraction logic is hidden on the server side.
    Clients only send base64 image and markdown text.

    Args:
        request: ProductExtractionRequest with image_base64 and markdown

    Returns:
        ProductExtractionResponse with product_offers list
    """
    temp_file_path = None

    try:
        # Validate markdown is provided
        if not request.markdown:
            raise HTTPException(status_code=400, detail="Markdown text is required")

        logger.info("Product extraction request received")

        # Convert base64 to PIL Image
        image = base64_to_image(request.image_base64)
        logger.info(f"Image converted successfully: {image.size}, mode: {image.mode}")

        # Save image to temporary file (extract_products_hybrid expects a file path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            image.save(temp_file, format="PNG")
            temp_file_path = temp_file.name

        logger.info(f"Saved image to temporary file: {temp_file_path}")

        # Import here to avoid circular dependency
        from inference.product_extraction.dspy_extraction import extract_products_hybrid

        # Run the hidden extraction pipeline
        result = extract_products_hybrid(image_path=temp_file_path, markdown=request.markdown)

        logger.info(f"Extraction complete: {len(result.get('product_offers', []))} product offers")

        return ProductExtractionResponse(product_offers=result.get('product_offers', []))

    except base64.binascii.Error as e:
        logger.error(f"Invalid base64 encoding: {e}")
        raise HTTPException(status_code=400, detail="Invalid image data")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Product extraction error: {e}")
        raise HTTPException(status_code=500, detail="Product extraction failed")
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_file_path}: {e}")


@app.post("/detect-company", response_model=CompanyDetectionResponse)
async def detect_company(request: CompanyDetectionRequest):
    """
    Detect company from image using internal VLM.

    The actual VLM call and prompts are hidden on the server side.
    Clients only send base64 image and receive company name.

    Args:
        request: CompanyDetectionRequest with image_base64

    Returns:
        CompanyDetectionResponse with detected company
    """
    try:
        logger.info("Company detection request received")

        # Convert base64 to PIL Image
        image = base64_to_image(request.image_base64)

        # Convert to JPEG base64 for VLM
        b64_image = image_to_jpeg_base64(image)

        # Create payload with hidden prompts
        payload = {
            "model": VLM_MODEL,
            "temperature": VLM_TEMPERATURE,
            "max_tokens": VLM_MAX_TOKENS,
            "messages": [
                {
                    "role": "system",
                    "content": COMPANY_IDENTIFICATION_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": COMPANY_RECOGNITION_SHORT_USER_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                        },
                    ],
                },
            ],
        }

        # Send request to VLM
        response = requests.post(VLM_URL, json=payload, timeout=60)

        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            # Clean JSON response (remove markdown code blocks)
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            logger.info(f"Company detection successful: {content}")
            return CompanyDetectionResponse(company=content)
        else:
            logger.error(f"VLM request failed: {response.status_code} - {response.text}")
            raise HTTPException(status_code=500, detail="Company detection failed")

    except base64.binascii.Error as e:
        logger.error(f"Invalid base64 encoding: {e}")
        raise HTTPException(status_code=400, detail="Invalid image data")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Company detection error: {e}")
        raise HTTPException(status_code=500, detail="Company detection failed")


@app.post("/map-product", response_model=ProductMappingResponse)
async def map_product(request: ProductMappingRequest):
    """
    Map product name to category pairs using internal LLM.

    The actual LLM call and prompts are hidden on the server side.
    Clients only send product name and category pairs, receive matched pairs.

    Args:
        request: ProductMappingRequest with product_name and category_pairs

    Returns:
        ProductMappingResponse with matched category pairs
    """
    try:
        logger.info(f"Product mapping request received for: {request.product_name}")

        # Format category pairs for the prompt
        pairs_text = "\n".join([
            f"- {pair[0]}: {pair[1]}" if len(pair) == 2 else f"- {pair[0]}"
            for pair in request.category_pairs
        ])

        # Generate prompt using hidden function
        user_prompt = get_product_mapping_prompt(request.product_name, pairs_text)

        # Create payload with hidden prompts
        payload = {
            "model": LLM_MODEL,
            "temperature": 0.1,
            "max_tokens": 1000,
            "messages": [
                {"role": "system", "content": PRODUCT_MAPPING_LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
        }

        # Send request to LLM
        response = requests.post(f"{LLM_URL}/chat/completions", json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            llm_response = result["choices"][0]["message"]["content"]

            # Parse JSON response
            try:
                # Clean response (remove markdown code blocks)
                llm_response = llm_response.strip()
                if llm_response.startswith("```json"):
                    llm_response = llm_response[7:]
                elif llm_response.startswith("```"):
                    llm_response = llm_response[3:]
                if llm_response.endswith("```"):
                    llm_response = llm_response[:-3]
                llm_response = llm_response.strip()

                parsed_response = json.loads(llm_response)
                logger.info(f"LLM returned {len(parsed_response)} matched categories")

                # Extract and validate category pairs
                matched_pairs = []
                valid_pairs_set = set((pair[0], pair[1]) if len(pair) == 2 else (pair[0], None) for pair in request.category_pairs)

                for item in parsed_response:
                    if isinstance(item, dict) and 'category' in item and 'type' in item:
                        pair = (item['category'], item['type'])
                        if pair in valid_pairs_set:
                            matched_pairs.append({"category": item['category'], "type": item['type']})

                # Return top 3
                return ProductMappingResponse(matched_pairs=matched_pairs[:3])

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {llm_response}, error: {e}")
                # Fallback: return empty result
                return ProductMappingResponse(matched_pairs=[])
        else:
            logger.error(f"LLM request failed: {response.status_code} - {response.text}")
            raise HTTPException(status_code=500, detail="Product mapping failed")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Product mapping error: {e}")
        raise HTTPException(status_code=500, detail="Product mapping failed")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Inference API server on port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8002)
