"""Image Restoration Module for Robustness and Error Decomposition Studies.

Provides standard, reproducible image restoration and visual enhancement methods:
1. Low-Light Enhancement:
   - CLAHE on luminance (LAB space)
   - Gamma curve boost
   - Retinex reflectance illumination decomposition
2. Dehazing:
   - Dark Channel Prior (DCP) atmospheric transmission recovery
   - Contrast enhancement
3. Denoising:
   - Bilateral edge-preserving filtering
   - Gaussian smoothing
   - Median filtering

Also evaluates the scientific trade-off:
Does restoration recover missed detections at the cost of increasing background false positives?
"""

from typing import Union, Optional, Tuple, Dict, Any, Callable
from pathlib import Path
import json
import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter

from degradations import _to_numpy_rgb, _from_numpy_rgb, apply_degradation
from error_analysis import decompose_errors
from pipeline import compare_error_profiles, run_experiment_condition, compute_coco_map


def enhance_low_light(
    image: Union[np.ndarray, Image.Image],
    method: str = "clahe",
    clip_limit: float = 2.5,
    tile_grid_size: Tuple[int, int] = (8, 8),
    gamma: float = 0.5,
) -> Union[np.ndarray, Image.Image]:
    """
    Enhances underexposed / low-light images.

    Args:
        image: RGB input image (PIL Image or uint8 NumPy array).
        method: Enhancement algorithm ('clahe', 'gamma', 'retinex').
        clip_limit: Contrast threshold for CLAHE.
        tile_grid_size: Grid division for adaptive equalization.
        gamma: Exponent for gamma correction (< 1.0 brightens shadows).

    Returns:
        Enhanced RGB image matching input type.
    """
    image_np, was_pil = _to_numpy_rgb(image)

    if method == "clahe":
        # Convert RGB -> LAB to operate purely on Luminance (L) channel
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced_l = clahe.apply(l_channel)

        enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
        enhanced_np = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    elif method == "gamma":
        # Power-law transform: I_out = 255 * (I / 255) ** gamma
        inv_gamma = float(gamma)
        norm = image_np.astype(np.float32) / 255.0
        enhanced_np = np.clip(np.power(norm, inv_gamma) * 255.0, 0, 255).astype(np.uint8)

    elif method == "retinex":
        # Single-scale Retinex: R(x, y) = log(I(x, y)) - log(G(sigma) * I(x, y))
        img_float = image_np.astype(np.float32) + 1.0  # Avoid log(0)
        # Spatial blur to estimate illumination component
        illumination = gaussian_filter(img_float, sigma=(15.0, 15.0, 0))
        retinex = np.log(img_float) - np.log(illumination)
        # Normalize back to [0, 255]
        mean = np.mean(retinex)
        std = np.std(retinex)
        if std > 0:
            norm_retinex = (retinex - (mean - 2.0 * std)) / (4.0 * std)
        else:
            norm_retinex = retinex
        enhanced_np = np.clip(norm_retinex * 255.0, 0, 255).astype(np.uint8)

    else:
        raise ValueError(f"Unknown low-light enhancement method '{method}'. Available: ['clahe', 'gamma', 'retinex']")

    return _from_numpy_rgb(enhanced_np, was_pil)


def dehaze(
    image: Union[np.ndarray, Image.Image],
    method: str = "dcp",
    omega: float = 0.85,
    patch_size: int = 15,
    t0: float = 0.1,
) -> Union[np.ndarray, Image.Image]:
    """
    Dehazes atmospheric haze/fog using Dark Channel Prior (DCP) or localized contrast enhancement.

    Args:
        image: RGB input image.
        method: 'dcp' (Dark Channel Prior) or 'clahe'.
        omega: Haze removal fraction (default 0.85 retains realistic perspective depth).
        patch_size: Patch window for dark channel calculation.
        t0: Minimum transmission lower bound to prevent noise over-amplification in dense haze.

    Returns:
        Dehazed RGB image matching input type.
    """
    image_np, was_pil = _to_numpy_rgb(image)

    if method == "dcp":
        img_float = image_np.astype(np.float32) / 255.0

        # Compute dark channel across min color channel in local window
        min_channel = np.min(img_float, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
        dark_channel = cv2.erode(min_channel, kernel)

        # Estimate atmospheric light A from brightest pixels in dark channel (top 0.1%)
        num_pixels = dark_channel.size
        top_k = max(1, int(num_pixels * 0.001))
        flat_dark = dark_channel.ravel()
        indices = np.argpartition(flat_dark, -top_k)[-top_k:]

        flat_img = img_float.reshape(-1, 3)
        atmospheric_light = np.mean(flat_img[indices], axis=0)
        atmospheric_light = np.clip(atmospheric_light, 0.1, 1.0)

        # Estimate transmission map t(x)
        norm_img = img_float / atmospheric_light
        min_norm = np.min(norm_img, axis=2)
        dark_norm = cv2.erode(min_norm, kernel)
        transmission = 1.0 - omega * dark_norm
        transmission = np.clip(transmission, t0, 1.0)

        # Soft guided filter / smooth transmission
        transmission = cv2.GaussianBlur(transmission, (patch_size * 2 + 1, patch_size * 2 + 1), 0)
        transmission = np.maximum(transmission, t0)

        # Recover scene radiance: J(x) = (I(x) - A) / max(t(x), t0) + A
        dehazed = np.zeros_like(img_float)
        for c in range(3):
            dehazed[:, :, c] = (img_float[:, :, c] - atmospheric_light[c]) / transmission + atmospheric_light[c]

        dehazed_np = np.clip(dehazed * 255.0, 0, 255).astype(np.uint8)

    elif method == "clahe":
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l)
        dehazed_lab = cv2.merge([enhanced_l, a, b])
        dehazed_np = cv2.cvtColor(dehazed_lab, cv2.COLOR_LAB2RGB)

    else:
        raise ValueError(f"Unknown dehazing method '{method}'. Available: ['dcp', 'clahe']")

    return _from_numpy_rgb(dehazed_np, was_pil)


def denoise(
    image: Union[np.ndarray, Image.Image],
    method: str = "bilateral",
    d: int = 7,
    sigma_color: float = 50.0,
    sigma_space: float = 50.0,
) -> Union[np.ndarray, Image.Image]:
    """
    Applies image denoising.

    Args:
        image: RGB input image.
        method: 'bilateral' (edge-preserving), 'gaussian', or 'median'.
        d: Diameter of pixel neighborhood for bilateral filtering.
        sigma_color: Filter sigma in color space.
        sigma_space: Filter sigma in coordinate space.

    Returns:
        Denoised RGB image matching input type.
    """
    image_np, was_pil = _to_numpy_rgb(image)

    if method == "bilateral":
        # Bilateral filter preserves sharp edges while smoothing sensor noise
        denoised_np = cv2.bilateralFilter(image_np, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    elif method == "gaussian":
        denoised_np = cv2.GaussianBlur(image_np, (d if d % 2 == 1 else d + 1, d if d % 2 == 1 else d + 1), sigma_space / 25.0)
    elif method == "median":
        ksize = d if d % 2 == 1 else d + 1
        denoised_np = cv2.medianBlur(image_np, ksize)
    else:
        raise ValueError(f"Unknown denoising method '{method}'. Available: ['bilateral', 'gaussian', 'median']")

    return _from_numpy_rgb(denoised_np, was_pil)


RESTORATION_REGISTRY: Dict[str, Callable[..., Union[np.ndarray, Image.Image]]] = {
    "low_light_enhancement": enhance_low_light,
    "dehazing": dehaze,
    "denoising": denoise,
}

AVAILABLE_RESTORATIONS = list(RESTORATION_REGISTRY.keys())


def apply_restoration(
    image: Union[np.ndarray, Image.Image],
    restoration_type: str,
    **kwargs: Any,
) -> Union[np.ndarray, Image.Image]:
    """Dispatches restoration method by name."""
    if restoration_type not in RESTORATION_REGISTRY:
        raise ValueError(f"Unknown restoration '{restoration_type}'. Available: {AVAILABLE_RESTORATIONS}")
    return RESTORATION_REGISTRY[restoration_type](image, **kwargs)


def restore_directory(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    restoration_type: str,
    **kwargs: Any,
) -> int:
    """Restores all images in input_dir and saves them to output_dir with identical filenames."""
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    supported_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted([p for p in in_path.iterdir() if p.suffix.lower() in supported_exts])

    if not image_paths:
        return 0

    for p in image_paths:
        with Image.open(p) as img:
            rgb_img = img.convert("RGB")
            restored = apply_restoration(rgb_img, restoration_type=restoration_type, **kwargs)
            save_path = out_path / p.name
            restored.save(save_path, quality=95)

    return len(image_paths)


def run_restoration_tradeoff_study(
    clean_images_dir: Union[str, Path],
    gt_source: Union[str, Path, Dict[str, Any]],
    model_name: str,
    degradation_type: str,
    severity: int,
    restoration_type: str,
    output_dir: Union[str, Path] = "results/restoration_study",
    restoration_kwargs: Optional[Dict[str, Any]] = None,
    detector_fn: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
    config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes a triad evaluation (Clean vs. Degraded vs. Restored) to quantify
    the diagnostic error trade-off introduced by restoration methods.

    Returns:
        Structured dictionary containing Clean, Degraded, and Restored error profiles,
        metric deltas, and tradeoff diagnosis.
    """
    out_base = Path(output_dir)
    clean_dir = Path(clean_images_dir)
    cfg = config or {}
    r_kwargs = restoration_kwargs or {}

    print(f"\n=======================================================")
    print(f"RESTORATION TRADEOFF STUDY")
    print(f"Model: {model_name} | Degradation: {degradation_type} (Sev {severity}) | Restoration: {restoration_type}")
    print(f"=======================================================\n")

    # Step 1: Clean Baseline
    print(">>> Condition 1: Clean Baseline...")
    clean_pred_path = out_base / "preds" / f"{model_name}_clean.json"
    clean_res = run_experiment_condition(
        images_dir=clean_dir,
        gt_source=gt_source,
        output_pred_path=clean_pred_path,
        model_name=model_name,
        config=cfg,
        detector_fn=detector_fn,
    )

    # Step 2: Degraded Condition
    print(f">>> Condition 2: Degraded ({degradation_type} Sev {severity})...")
    deg_img_dir = out_base / "images" / f"{degradation_type}_sev{severity}"
    from degradations import degrade_directory
    degrade_directory(clean_dir, deg_img_dir, degradation_type, severity=severity, seed=seed)

    deg_pred_path = out_base / "preds" / f"{model_name}_{degradation_type}_sev{severity}.json"
    deg_res = run_experiment_condition(
        images_dir=deg_img_dir,
        gt_source=gt_source,
        output_pred_path=deg_pred_path,
        model_name=model_name,
        config=cfg,
        detector_fn=detector_fn,
    )

    # Step 3: Restored Condition
    print(f">>> Condition 3: Restored ({restoration_type})...")
    rest_img_dir = out_base / "images" / f"restored_{restoration_type}_{degradation_type}_sev{severity}"
    restore_directory(deg_img_dir, rest_img_dir, restoration_type=restoration_type, **r_kwargs)

    rest_pred_path = out_base / "preds" / f"{model_name}_restored_{restoration_type}.json"
    rest_res = run_experiment_condition(
        images_dir=rest_img_dir,
        gt_source=gt_source,
        output_pred_path=rest_pred_path,
        model_name=model_name,
        config=cfg,
        detector_fn=detector_fn,
    )

    # Compute comparative shifts
    deg_raw = deg_res["raw_counts"]
    rest_raw = rest_res["raw_counts"]

    # Trade-off metrics: Degraded -> Restored
    misses_recovered = deg_raw["miss"] - rest_raw["miss"]  # Positive means fewer missed objects!
    bkg_fp_introduced = rest_raw["bkg"] - deg_raw["bkg"]   # Positive means more background hallucinations!
    loc_err_shift = rest_raw["loc"] - deg_raw["loc"]

    tradeoff_diagnosis = (
        f"Restoration impact: Missed objects change = {-misses_recovered:+d} "
        f"({'recovered' if misses_recovered > 0 else 'added'}), "
        f"Background False Positives change = {bkg_fp_introduced:+d} "
        f"({'hallucinated' if bkg_fp_introduced > 0 else 'reduced'})."
    )

    study_summary = {
        "model": model_name,
        "degradation": degradation_type,
        "severity": severity,
        "restoration": restoration_type,
        "clean_baseline": clean_res["summary"],
        "degraded": deg_res["summary"],
        "restored": rest_res["summary"],
        "tradeoff_metrics": {
            "misses_recovered": int(misses_recovered),
            "background_fp_introduced": int(bkg_fp_introduced),
            "localization_error_shift": int(loc_err_shift),
            "degraded_miss_rate": deg_res["summary"]["miss_rate"],
            "restored_miss_rate": rest_res["summary"]["miss_rate"],
            "degraded_bkg_fp": deg_raw["bkg"],
            "restored_bkg_fp": rest_raw["bkg"],
        },
        "tradeoff_diagnosis": tradeoff_diagnosis,
    }

    out_file = out_base / f"{model_name}_{degradation_type}_sev{severity}_{restoration_type}_tradeoff.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(study_summary, f, indent=2)

    print(f"\nTradeoff Result: {tradeoff_diagnosis}")
    print(f"Summary saved to: {out_file}\n")

    return study_summary
