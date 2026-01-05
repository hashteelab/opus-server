"""
Image preprocessing utilities for improving VLM recognition.

This module provides functions to preprocess images before they are sent to
Vision Language Models (VLMs) for better text extraction and recognition.
Preprocessing includes converting to grayscale and enhancing contrast.
"""

import logging
from PIL import Image, ImageEnhance
from typing import Union
import io
import base64
from io import BytesIO

# Configure logging
from logger import logger


def convert_to_grayscale(image: Image.Image) -> Image.Image:
    """
    Convert a PIL Image to grayscale.
    
    Args:
        image (Image.Image): Input PIL Image object
        
    Returns:
        Image.Image: Grayscale version of the input image
        
    Example:
        >>> from PIL import Image
        >>> img = Image.open("sample.jpg")
        >>> gray_img = convert_to_grayscale(img)
    """
    try:
        # Convert to grayscale ('L' mode)
        grayscale_image = image.convert('L')
        logger.info("Image converted to grayscale")
        return grayscale_image
    except Exception as e:
        logger.error(f"Failed to convert image to grayscale: {e}")
        raise Exception(f"Grayscale conversion failed: {e}")


def enhance_contrast(image: Image.Image, factor: float = 2.0) -> Image.Image:
    """
    Enhance the contrast of a PIL Image.
    
    Args:
        image (Image.Image): Input PIL Image object
        factor (float): Contrast enhancement factor. 
                       factor < 1.0 decreases contrast
                       factor = 1.0 keeps original contrast
                       factor > 1.0 increases contrast
                       Recommended range: 1.5 - 3.0
                       
    Returns:
        Image.Image: Contrast-enhanced version of the input image
        
    Example:
        >>> from PIL import Image
        >>> img = Image.open("sample.jpg")
        >>> enhanced_img = enhance_contrast(img, factor=2.5)
    """
    try:
        # Create a contrast enhancer
        enhancer = ImageEnhance.Contrast(image)
        
        # Apply contrast enhancement
        enhanced_image = enhancer.enhance(factor)
        logger.info(f"Image contrast enhanced with factor: {factor}")
        return enhanced_image
    except Exception as e:
        logger.error(f"Failed to enhance image contrast: {e}")
        raise Exception(f"Contrast enhancement failed: {e}")


def preprocess_image_for_vlm(
    image: Image.Image, 
    grayscale: bool = True, 
    enhance_contrast_flag: bool = True,
    contrast_factor: float = 2.0
) -> Image.Image:
    """
    Apply complete preprocessing pipeline to an image for VLM processing.
    
    This function combines grayscale conversion and contrast enhancement
    to prepare images for better text extraction by Vision Language Models.
    
    Args:
        image (Image.Image): Input PIL Image object
        grayscale (bool): Whether to convert image to grayscale. Default: True
        enhance_contrast_flag (bool): Whether to enhance contrast. Default: True
        contrast_factor (float): Contrast enhancement factor. Default: 2.0
        
    Returns:
        Image.Image: Preprocessed image ready for VLM processing
        
    Example:
        >>> from PIL import Image
        >>> img = Image.open("document.jpg")
        >>> processed_img = preprocess_image_for_vlm(img, contrast_factor=2.5)
    """
    try:
        processed_image = image.copy()  # Work on a copy to preserve original
        
        # Step 1: Convert to grayscale if requested
        if grayscale:
            processed_image = convert_to_grayscale(processed_image)
            logger.info("Preprocessing: Grayscale conversion applied")
        
        # Step 2: Enhance contrast if requested
        if enhance_contrast_flag:
            processed_image = enhance_contrast(processed_image, factor=contrast_factor)
            logger.info(f"Preprocessing: Contrast enhancement applied (factor={contrast_factor})")
        
        logger.info("Image preprocessing completed successfully")
        return processed_image
        
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise Exception(f"Preprocessing pipeline failed: {e}")


def preprocess_image_from_path(
    image_path: str,
    grayscale: bool = True,
    enhance_contrast_flag: bool = True,
    contrast_factor: float = 2.0,
    save_path: str = None
) -> Image.Image:
    """
    Load an image from file path, preprocess it, and optionally save the result.
    
    Args:
        image_path (str): Path to the input image file
        grayscale (bool): Whether to convert image to grayscale. Default: True
        enhance_contrast_flag (bool): Whether to enhance contrast. Default: True
        contrast_factor (float): Contrast enhancement factor. Default: 2.0
        save_path (str, optional): Path to save the preprocessed image. 
                                   If None, image is not saved. Default: None
        
    Returns:
        Image.Image: Preprocessed PIL Image object
        
    Example:
        >>> processed_img = preprocess_image_from_path(
        ...     "input.jpg", 
        ...     contrast_factor=2.5,
        ...     save_path="output.jpg"
        ... )
    """
    try:
        # Load the image
        image = Image.open(image_path)
        logger.info(f"Image loaded from: {image_path}")
        
        # Apply preprocessing
        processed_image = preprocess_image_for_vlm(
            image, 
            grayscale=grayscale,
            enhance_contrast_flag=enhance_contrast_flag,
            contrast_factor=contrast_factor
        )
        
        # Save if path is provided
        if save_path:
            processed_image.save(save_path)
            logger.info(f"Preprocessed image saved to: {save_path}")
        
        return processed_image
        
    except Exception as e:
        logger.error(f"Failed to preprocess image from path: {e}")
        raise Exception(f"Image preprocessing from path failed: {e}")


def apply_adaptive_contrast(image: Image.Image) -> Image.Image:
    """
    Apply adaptive contrast enhancement based on image characteristics.
    
    This function analyzes the image and applies an appropriate contrast
    enhancement factor automatically.
    
    Args:
        image (Image.Image): Input PIL Image object
        
    Returns:
        Image.Image: Adaptively contrast-enhanced image
    """
    try:
        # Convert to grayscale for analysis if needed
        if image.mode != 'L':
            gray = image.convert('L')
        else:
            gray = image
        
        # Calculate histogram statistics
        histogram = gray.histogram()
        pixels = sum(histogram)
        
        # Calculate mean brightness
        mean_brightness = sum(i * histogram[i] for i in range(256)) / pixels
        
        # Determine contrast factor based on brightness
        # Darker images need more contrast enhancement
        if mean_brightness < 85:
            factor = 2.5  # Low brightness - high contrast
        elif mean_brightness < 170:
            factor = 2.0  # Medium brightness - medium contrast
        else:
            factor = 1.5  # High brightness - lower contrast
        
        logger.info(f"Adaptive contrast: mean_brightness={mean_brightness:.1f}, factor={factor}")
        
        # Apply the calculated contrast factor
        return enhance_contrast(image, factor=factor)
        
    except Exception as e:
        logger.error(f"Adaptive contrast enhancement failed: {e}")
        raise Exception(f"Adaptive contrast failed: {e}")


def PILimage_to_base64(image: Image.Image, format: str = 'PNG') -> str:
    """
    Convert a PIL Image to base64 encoded string with data URI format.
    
    This function is specifically designed for DotsOCR API which expects
    base64 encoded images in data URI format.
    
    Args:
        image (Image.Image): Input PIL Image object
        format (str): Image format for encoding. Default: 'PNG'
                     Supported formats: 'PNG', 'JPEG', 'JPG', etc.
        
    Returns:
        str: Base64 encoded image string in data URI format
             Format: "data:image/{format};base64,{base64_string}"
        
    Example:
        >>> from PIL import Image
        >>> img = Image.open("sample.jpg")
        >>> base64_str = PILimage_to_base64(img, format='PNG')
        >>> # Output: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
    """
    try:
        buffered = BytesIO()
        image.save(buffered, format=format)
        base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        result = f"data:image/{format.lower()};base64,{base64_str}"
        logger.info(f"PIL Image converted to base64 (format: {format})")
        return result
        
    except Exception as e:
        logger.error(f"Failed to convert PIL image to base64: {e}")
        raise Exception(f"PIL to base64 conversion failed: {e}")
