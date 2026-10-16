"""Image Degradation Module for Robustness and Error Decomposition Studies.

Provides controlled, reproducible visual degradations at standardized severity levels (1 to 5).
Each degradation preserves image dimensions, RGB channel ordering, and uint8 dynamic range [0, 255].
"""

from typing import Union, Optional, Tuple, Dict, Any, Callable
from pathlib import Path
import numpy as np
from PIL import Image


# Severity parameters for Low Light degradation.
# Combines gamma correction (simulating non-linear camera response and dynamic range loss)
# with linear luminance attenuation factor (beta).
# Severity 1: Mild dusk/shadow (gamma=1.3, beta=0.80)
# Severity 2: Indoor dim light (gamma=1.6, beta=0.60)
# Severity 3: Low light / street twilight (gamma=2.0, beta=0.40)
# Severity 4: Dark night scene (gamma=2.5, beta=0.25)
# Severity 5: Extreme low light (gamma=3.2, beta=0.12)
LOW_LIGHT_PARAMS: Dict[int, Dict[str, float]] = {
    1: {"gamma": 1.3, "beta": 0.80},
    2: {"gamma": 1.6, "beta": 0.60},
    3: {"gamma": 2.0, "beta": 0.40},
    4: {"gamma": 2.5, "beta": 0.25},
    5: {"gamma": 3.2, "beta": 0.12},
}


def _validate_severity(severity: int) -> None:
    """Validates that severity is an integer between 1 and 5."""
    if not isinstance(severity, int) or severity < 1 or severity > 5:
        raise ValueError(
            f"Severity must be an integer between 1 and 5, got {severity} (type: {type(severity).__name__})"
        )


def _to_numpy_rgb(image: Union[np.ndarray, Image.Image]) -> Tuple[np.ndarray, bool]:
    """
    Validates and converts input image to an RGB uint8 NumPy array.

    Returns:
        (image_np, was_pil): tuple containing the RGB ndarray and a boolean
        indicating whether the original input was a PIL Image.
    """
    was_pil = False
    if isinstance(image, Image.Image):
        was_pil = True
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_np = np.array(image, dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            raise ValueError(f"NumPy image array must have dtype uint8, got {image.dtype}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected 3-channel (H, W, 3) RGB image, got shape {image.shape}"
            )
        # Defensive copy to ensure input array is never modified in-place
        image_np = image.copy()
    else:
        raise TypeError(
            f"Expected image to be PIL.Image.Image or np.ndarray, got {type(image).__name__}"
        )

    return image_np, was_pil


def _from_numpy_rgb(image_np: np.ndarray, to_pil: bool) -> Union[np.ndarray, Image.Image]:
    """Converts RGB uint8 NumPy array back to PIL Image if required."""
    image_clipped = np.clip(image_np, 0, 255).astype(np.uint8)
    if to_pil:
        return Image.fromarray(image_clipped, mode="RGB")
    return image_clipped


def apply_low_light(
    image: Union[np.ndarray, Image.Image],
    severity: int = 1,
    seed: Optional[int] = None,
) -> Union[np.ndarray, Image.Image]:
    """
    Applies controlled low-light degradation to an RGB image.

    Operational definition:
        Non-linear gamma adjustment and linear attenuation:
        I_out = beta * (I_in / 255.0) ** gamma * 255.0

    Args:
        image: Input RGB image as PIL Image or uint8 NumPy array (H, W, 3).
        severity: Degradation level from 1 (mild) to 5 (extreme).
        seed: Random seed for deterministic reproducibility if stochastic elements are present.

    Returns:
        Degraded RGB image in the same type as input (PIL Image or uint8 np.ndarray).
    """
    _validate_severity(severity)
    image_np, was_pil = _to_numpy_rgb(image)

    params = LOW_LIGHT_PARAMS[severity]
    gamma = params["gamma"]
    beta = params["beta"]

    # Normalize to [0, 1] float
    norm = image_np.astype(np.float32) / 255.0

    # Non-linear tone scaling
    degraded_norm = beta * np.power(norm, gamma)

    # Scale back to [0, 255]
    degraded_np = degraded_norm * 255.0

    return _from_numpy_rgb(degraded_np, was_pil)


DEGRADATION_REGISTRY: Dict[str, Callable[..., Union[np.ndarray, Image.Image]]] = {
    "low_light": apply_low_light,
}

AVAILABLE_DEGRADATIONS = list(DEGRADATION_REGISTRY.keys())


def apply_degradation(
    image: Union[np.ndarray, Image.Image],
    degradation_type: str,
    severity: int = 1,
    seed: Optional[int] = None,
) -> Union[np.ndarray, Image.Image]:
    """
    Dispatches degradation application based on type and severity.

    Args:
        image: Input RGB image (PIL Image or uint8 np.ndarray).
        degradation_type: Registered degradation name (e.g., 'low_light').
        severity: Integer from 1 to 5.
        seed: Optional random seed.

    Returns:
        Degraded RGB image of the same type as input.
    """
    if degradation_type not in DEGRADATION_REGISTRY:
        raise ValueError(
            f"Unknown degradation type '{degradation_type}'. "
            f"Available types: {AVAILABLE_DEGRADATIONS}"
        )

    transform_fn = DEGRADATION_REGISTRY[degradation_type]
    return transform_fn(image, severity=severity, seed=seed)


def degrade_directory(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    degradation_type: str,
    severity: int = 1,
    seed: Optional[int] = 42,
) -> int:
    """
    Applies degradation to all images in input_dir and saves degraded images to output_dir.

    Args:
        input_dir: Path to directory containing source images.
        output_dir: Path to destination directory.
        degradation_type: Degradation name (e.g. 'low_light').
        severity: Severity level (1-5).
        seed: Random seed for reproducibility.

    Returns:
        Number of images processed.
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    supported_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(
        [p for p in in_path.iterdir() if p.suffix.lower() in supported_exts]
    )

    if not image_paths:
        return 0

    for idx, p in enumerate(image_paths):
        # Per-image deterministic seed
        img_seed = None if seed is None else seed + idx
        with Image.open(p) as img:
            rgb_img = img.convert("RGB")
            degraded = apply_degradation(
                rgb_img,
                degradation_type=degradation_type,
                severity=severity,
                seed=img_seed,
            )
            # Save using identical filename to maintain 1:1 pairing
            save_path = out_path / p.name
            degraded.save(save_path, quality=95)

    return len(image_paths)
