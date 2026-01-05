"""
VLM (Vision Language Model) Client for image-based extraction.

This module provides a client for interacting with vision language models,
handling image encoding, API requests with retry logic, and response parsing.
"""

import base64
import requests
import json
from PIL import Image
import io
from typing import Optional, Dict
from .image_utils import preprocess_image_for_vlm
from logger import log_success, log_processing, log_warning, log_info


class VLMClient:
    """
    Client for Vision Language Model API requests.

    Handles image preprocessing, encoding, payload creation, and HTTP requests
    with retry logic for robust VLM interactions.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize VLMClient with configuration.

        Args:
            config: Configuration dictionary. If None, uses default from models_config.
                    Expected keys: url, model, temperature, max_tokens, request_timeout,
                    max_retries, enable_preprocessing, convert_to_grayscale, enhance_contrast,
                    contrast_factor, max_image_dimension, max_encoded_size_mb, jpeg_quality
        """
        if config is None:
            # Lazy import to avoid circular dependency
            from inference import get_vlm_config, VLMProviderEnum
            config = get_vlm_config(VLMProviderEnum.qwen3_vl)

        # VLM Configuration
        self.url = config["url"]
        self.model = config["model"]
        self.temperature = config["temperature"]
        self.max_tokens = config["max_tokens"]
        self.request_timeout = config["request_timeout"]
        self.max_retries = config["max_retries"]

        # Image Configuration
        self.enable_preprocessing = config["enable_preprocessing"]
        self.convert_to_grayscale = config["convert_to_grayscale"]
        self.enhance_contrast = config["enhance_contrast"]
        self.contrast_factor = config["contrast_factor"]
        self.max_image_dimension = config["max_image_dimension"]
        self.max_encoded_size_mb = config["max_encoded_size_mb"]
        self.jpeg_quality = config["jpeg_quality"]

    # -----------------------------------------
    # Step 1: Encode image -> base64 string
    # -----------------------------------------
    def _encode_image_to_base64(self, file_path: str) -> str:
        """
        Encode image to base64 with validation and optional preprocessing

        Args:
            file_path: Path to the image file

        Returns:
            Base64 encoded string

        Raises:
            Exception: If encoding fails or image is too large
        """
        try:
            image = Image.open(file_path)
            original_size = image.size

            # Resize if image is too large
            if max(image.size) > self.max_image_dimension:
                image.thumbnail(
                    (self.max_image_dimension, self.max_image_dimension),
                    Image.Resampling.LANCZOS
                )
                log_info(f"🔧 Image resized from {original_size} to {image.size}", indent=0)

            # Apply preprocessing if enabled
            if self.enable_preprocessing:
                image = preprocess_image_for_vlm(
                    image,
                    grayscale=self.convert_to_grayscale,
                    enhance_contrast_flag=self.enhance_contrast,
                    contrast_factor=self.contrast_factor
                )

                # Convert back to RGB if grayscale was applied
                if image.mode == 'L':
                    image = image.convert('RGB')
            else:
                # Ensure RGB mode even without preprocessing
                if image.mode not in ('RGB', 'RGBA'):
                    image = image.convert('RGB')

            # Convert to JPEG and encode
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='JPEG', quality=self.jpeg_quality)
            img_buffer.seek(0)

            encoded = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
            img_buffer.close()

            # Validate encoded size
            size_mb = len(encoded) / (1024 * 1024)
            if size_mb > self.max_encoded_size_mb:
                raise Exception(
                    f"Encoded image too large: {size_mb:.2f}MB "
                    f"(max: {self.max_encoded_size_mb}MB). "
                    f"Try reducing max_image_dimension or jpeg_quality."
                )

            print(f"✅ Image encoded: {size_mb:.2f}MB")
            return encoded

        except FileNotFoundError:
            raise Exception(f"Image file not found: {file_path}")
        except Exception as e:
            raise Exception(f"Failed to encode image to base64: {str(e)}")

    # -----------------------------------------
    # Step 2: Create payload
    # -----------------------------------------
    def _create_payload(self, user_prompt: str, b64_image: str) -> dict:
        """
        Create OpenAI-compatible payload for vLLM

        Args:
            user_prompt: The text prompt for the model
            b64_image: Base64 encoded image string

        Returns:
            Dictionary payload for API request
        """
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                        }
                    ]
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

    # -----------------------------------------
    # Step 3: Send request with retry logic
    # -----------------------------------------
    def _send_request(self, payload: dict) -> dict:
        """
        Send request to vLLM endpoint with retry logic

        Args:
            payload: Request payload dictionary

        Returns:
            Response JSON dictionary

        Raises:
            Exception: If request fails after all retries
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                res = requests.post(
                    self.url,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(payload),
                    timeout=self.request_timeout
                )

                # Check for HTTP errors
                res.raise_for_status()

                # Parse and return response
                response_data = res.json()
                return response_data

            except requests.exceptions.Timeout:
                last_error = f"Request timed out after {self.request_timeout}s"
                if attempt < self.max_retries - 1:
                    print(f"⚠️ {last_error}, retrying ({attempt + 1}/{self.max_retries})...")
                    continue

            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code
                error_body = e.response.text

                # Don't retry on client errors (4xx)
                if 400 <= status_code < 500:
                    raise Exception(
                        f"HTTP {status_code} Error: {error_body}\n"
                        f"This is likely a configuration or request format issue."
                    )

                # Retry on server errors (5xx)
                last_error = f"HTTP {status_code}: {error_body}"
                if attempt < self.max_retries - 1:
                    print(f"⚠️ Server error, retrying ({attempt + 1}/{self.max_retries})...")
                    continue

            except requests.exceptions.RequestException as e:
                last_error = f"Request failed: {str(e)}"
                if attempt < self.max_retries - 1:
                    print(f"⚠️ {last_error}, retrying ({attempt + 1}/{self.max_retries})...")
                    continue

            except json.JSONDecodeError as e:
                raise Exception(f"Invalid JSON response from server: {str(e)}")

        # If we've exhausted all retries
        raise Exception(f"Request failed after {self.max_retries} attempts. Last error: {last_error}")

    # -----------------------------------------
    # Public Method: Generic VLM query
    # -----------------------------------------
    def run(self, user_prompt: str, file_path: str) -> str:
        """
        Run a generic VLM query on an image

        Args:
            user_prompt: The prompt/question for the VLM
            file_path: Path to the image file

        Returns:
            VLM response as string

        Raises:
            Exception: If query fails
        """
        try:
            print(f"🔍 Processing image: {file_path}")
            b64_image = self._encode_image_to_base64(file_path)

            print(f"📤 Sending request to VLM...")
            payload = self._create_payload(user_prompt, b64_image)
            response = self._send_request(payload)

            print(f"✅ Received response from VLM")
            content = response["choices"][0]["message"]["content"]

            return content

        except Exception as e:
            raise Exception(f"VLM query failed: {str(e)}")
