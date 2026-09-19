from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from .contracts import ErrorCode, ImageLimits, ImageReport, ServiceError


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_image(source: Path, destination: Path, limits: ImageLimits) -> ImageReport:
    """Validate and normalize an uploaded image to metadata-free RGB PNG."""
    if not source.is_file():
        raise ServiceError(ErrorCode.IMAGE_NOT_FOUND, "Input image does not exist")

    size = source.stat().st_size
    if size > limits.max_file_bytes:
        raise ServiceError(
            ErrorCode.IMAGE_TOO_LARGE,
            f"Image is {size} bytes; limit is {limits.max_file_bytes} bytes",
        )

    try:
        with Image.open(source) as probe:
            source_format = (probe.format or "").upper()
            width, height = probe.size
            probe.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ServiceError(ErrorCode.INVALID_IMAGE, "Input is not a valid supported image") from exc

    if source_format not in limits.allowed_formats:
        raise ServiceError(
            ErrorCode.INVALID_IMAGE,
            f"Image format {source_format or 'unknown'} is not allowed",
        )
    if width > limits.max_width or height > limits.max_height:
        raise ServiceError(
            ErrorCode.IMAGE_DIMENSIONS_EXCEEDED,
            f"Image dimensions {width}x{height} exceed {limits.max_width}x{limits.max_height}",
        )

    try:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened)
            if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                image = Image.alpha_composite(background, rgba).convert("RGB")
            else:
                image = image.convert("RGB")
            destination.parent.mkdir(parents=True, exist_ok=True)
            image.save(destination, format="PNG", optimize=True)
    except (OSError, ValueError) as exc:
        raise ServiceError(ErrorCode.INVALID_IMAGE, "Image normalization failed") from exc

    return ImageReport(
        sha256=_sha256(source),
        source_format=source_format,
        width=width,
        height=height,
        normalized_filename=destination.name,
    )
