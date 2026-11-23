"""
Vision Module — Image Utilities
=================================
Helpers for reading, validating, and encoding images.
"""

import io
import base64
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("mdris.vision.image_utils")

# Supported MIME types → PIL format names
SUPPORTED_FORMATS = {
    "image/jpeg": "JPEG",
    "image/jpg": "JPEG",
    "image/png": "PNG",
}

# Max dimensions to prevent memory exhaustion
MAX_WIDTH = 1920
MAX_HEIGHT = 1080
MAX_FILE_BYTES = 15 * 1024 * 1024  # 15 MB


def validate_image_bytes(data: bytes, content_type: str) -> None:
    """
    Validate raw image bytes.

    Raises:
        ValueError: if format is unsupported or file is too large.
    """
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(
            f"Image too large ({len(data) // 1024} KB). Maximum is {MAX_FILE_BYTES // 1024} KB."
        )

    mime = content_type.lower().split(";")[0].strip()
    if mime not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported image format '{content_type}'. "
            f"Supported formats: {', '.join(SUPPORTED_FORMATS.keys())}"
        )


def open_image(data: bytes):
    """
    Open and optionally resize an image using PIL.

    Returns:
        PIL.Image.Image
    """
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError("Pillow is not installed. Run: pip install Pillow")

    img = Image.open(io.BytesIO(data)).convert("RGB")

    # Resize if needed to keep inference fast
    w, h = img.size
    if w > MAX_WIDTH or h > MAX_HEIGHT:
        img.thumbnail((MAX_WIDTH, MAX_HEIGHT), Image.LANCZOS)
        logger.debug(f"Image resized from ({w}×{h}) to {img.size}")

    return img


def encode_image_to_base64(image_path: str) -> Optional[str]:
    """
    Read a saved image file and encode it as a base64 string.

    Returns:
        Base64-encoded image string, or None if the file doesn't exist.
    """
    path = Path(image_path)
    if not path.exists():
        logger.warning(f"Annotated image not found at: {image_path}")
        return None
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as exc:
        logger.error(f"Failed to encode image: {exc}")
        return None


def save_pil_image(image, output_path: str) -> bool:
    """
    Save a PIL image to disk.

    Returns:
        True on success, False on failure.
    """
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        return True
    except Exception as exc:
        logger.error(f"Failed to save image to {output_path}: {exc}")
        return False
