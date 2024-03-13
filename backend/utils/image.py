"""Utility functions for image processing."""

import base64
import io

from PIL import Image


def resize_base64_image(
    base64_string: str, max_size: tuple[int, int] = (2000, 768)
) -> str:
    """Resize an image encoded as a Base64 string.

    Args:
        base64_string (str): Base64 string.
        max_size (tuple[int, int], optional): Maximum size of the image in the format
            (width, height). Defaults to (2000, 768).

    Returns:
        str: Re-sized Base64 string.
    """
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Resize the image keeping the aspect ratio
    img.thumbnail(max_size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
