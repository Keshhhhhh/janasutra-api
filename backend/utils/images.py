"""Memory-safe photo decoding for the models (the hosted API has 512 MB)."""
import io

from PIL import Image

# Every model works at 640 px or less; 1024 px on the short side keeps
# headroom for the authenticity crops.
MODEL_SIDE = 1024


def open_rgb(image_bytes: bytes, min_side: int = MODEL_SIDE) -> Image.Image:
    """Decode an upload as RGB, scaled down so its short side is about min_side.

    JPEG draft mode decodes straight at 1/2, 1/4 or 1/8 scale, so a 12 MP phone
    photo never exists as a full-size bitmap (~36 MB per copy).
    """
    img = Image.open(io.BytesIO(image_bytes))
    img.draft("RGB", (min_side, min_side))
    img = img.convert("RGB")
    short = min(img.size)
    if short > min_side * 1.5:
        s = min_side / short
        img = img.resize((round(img.width * s), round(img.height * s)), Image.BILINEAR)
    return img
