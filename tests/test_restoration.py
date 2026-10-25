import tempfile
from pathlib import Path
import json
import numpy as np
import pytest
from PIL import Image

from restoration import (
    enhance_low_light,
    dehaze,
    denoise,
    apply_restoration,
    restore_directory,
    run_restoration_tradeoff_study,
    AVAILABLE_RESTORATIONS,
)


@pytest.fixture
def dark_image():
    """Synthetic dark/underexposed image."""
    arr = np.full((64, 64, 3), 30, dtype=np.uint8)
    arr[20:40, 20:40, :] = 50
    return arr


@pytest.fixture
def hazy_image():
    """Synthetic low-contrast hazy image."""
    arr = np.full((64, 64, 3), 200, dtype=np.uint8)
    arr[20:40, 20:40, :] = 180
    return arr


@pytest.fixture
def noisy_image():
    """Synthetic noisy image."""
    rng = np.random.RandomState(42)
    base = np.full((64, 64, 3), 128, dtype=np.uint8)
    noise = rng.normal(0, 20, size=(64, 64, 3))
    return np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def test_registry_contains_restorations():
    expected = {"low_light_enhancement", "dehazing", "denoising"}
    assert expected.issubset(set(AVAILABLE_RESTORATIONS))


@pytest.mark.parametrize("method", ["clahe", "gamma", "retinex"])
def test_low_light_enhancement(dark_image, method):
    enhanced = enhance_low_light(dark_image, method=method)
    assert isinstance(enhanced, np.ndarray)
    assert enhanced.shape == dark_image.shape
    assert enhanced.dtype == np.uint8
    assert np.all(enhanced >= 0) and np.all(enhanced <= 255)

    # Enhanced image should be brighter than the dark input
    assert float(np.mean(enhanced)) > float(np.mean(dark_image))


def test_low_light_pil_input_preservation(dark_image):
    pil_dark = Image.fromarray(dark_image, mode="RGB")
    enhanced_pil = enhance_low_light(pil_dark, method="clahe")
    assert isinstance(enhanced_pil, Image.Image)
    assert enhanced_pil.size == pil_dark.size
    assert enhanced_pil.mode == "RGB"


@pytest.mark.parametrize("method", ["dcp", "clahe"])
def test_dehazing(hazy_image, method):
    dehazed = dehaze(hazy_image, method=method)
    assert isinstance(dehazed, np.ndarray)
    assert dehazed.shape == hazy_image.shape
    assert dehazed.dtype == np.uint8
    assert np.all(dehazed >= 0) and np.all(dehazed <= 255)


@pytest.mark.parametrize("method", ["bilateral", "gaussian", "median"])
def test_denoising(noisy_image, method):
    denoised = denoise(noisy_image, method=method)
    assert isinstance(denoised, np.ndarray)
    assert denoised.shape == noisy_image.shape
    assert denoised.dtype == np.uint8
    assert np.all(denoised >= 0) and np.all(denoised <= 255)

    # High frequency noise should be smoothed (lower variance across flat regions)
    assert float(np.std(denoised)) <= float(np.std(noisy_image))


def test_input_immutability(dark_image):
    copy_orig = dark_image.copy()
    _ = enhance_low_light(dark_image, method="clahe")
    np.testing.assert_array_equal(dark_image, copy_orig)


def test_invalid_restoration_type_raises_value_error(dark_image):
    with pytest.raises(ValueError, match="Unknown restoration"):
        apply_restoration(dark_image, restoration_type="non_existent")


def test_restore_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()

        img = Image.fromarray(np.full((32, 32, 3), 40, dtype=np.uint8), mode="RGB")
        img.save(in_dir / "sample.jpg")

        count = restore_directory(in_dir, out_dir, restoration_type="low_light_enhancement", method="gamma")
        assert count == 1
        assert (out_dir / "sample.jpg").is_file()


def test_restoration_tradeoff_study_end_to_end():
    """
    Tests the core research question:
    Validates whether restoration recovers missed detections while introducing background false alarms.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        clean_dir = tmp_path / "clean"
        clean_dir.mkdir()
        out_dir = tmp_path / "tradeoff_results"

        # Create two sample clean images
        img1 = Image.fromarray(np.full((64, 64, 3), 160, dtype=np.uint8), mode="RGB")
        img2 = Image.fromarray(np.full((64, 64, 3), 160, dtype=np.uint8), mode="RGB")
        img1.save(clean_dir / "101.jpg")
        img2.save(clean_dir / "102.jpg")

        # Ground truth
        gt_dict = {
            "annotations": [
                {"image_id": 101, "category_id": 1, "bbox": [10, 10, 20, 20], "id": 1},
                {"image_id": 102, "category_id": 1, "bbox": [15, 15, 25, 25], "id": 2},
            ]
        }

        # Mock detector modeling real-world trade-off behavior:
        # - Clean: detects both objects accurately.
        # - Degraded (dark): misses object on image 102 due to low light.
        # - Restored (brightened): recovers image 102, but introduces a background false positive
        #   on image 101 due to amplified noise artifacts!
        def mock_detector(images_path, pred_path, config):
            in_p = Path(images_path)
            folder_name = in_p.name
            preds = []

            for img_p in in_p.iterdir():
                if img_p.suffix == ".jpg":
                    img_id = int(img_p.stem)
                    gt_box = [10, 10, 20, 20] if img_id == 101 else [15, 15, 25, 25]

                    if "clean" in folder_name:
                        # Clean: perfect detections
                        preds.append({"image_id": img_id, "category_id": 1, "bbox": gt_box, "score": 0.95})
                    elif "low_light" in folder_name and "restored" not in folder_name:
                        # Degraded: misses object in 102
                        if img_id == 101:
                            preds.append({"image_id": 101, "category_id": 1, "bbox": gt_box, "score": 0.80})
                    elif "restored" in folder_name:
                        # Restored: recovers 102, but introduces background FP on 101!
                        preds.append({"image_id": img_id, "category_id": 1, "bbox": gt_box, "score": 0.88})
                        if img_id == 101:
                            # Hallucination / Background False Positive
                            preds.append({"image_id": 101, "category_id": 1, "bbox": [50, 50, 10, 10], "score": 0.70})

            with open(pred_path, "w") as f:
                json.dump(preds, f)

        study = run_restoration_tradeoff_study(
            clean_images_dir=clean_dir,
            gt_source=gt_dict,
            model_name="test_model",
            degradation_type="low_light",
            severity=3,
            restoration_type="low_light_enhancement",
            output_dir=out_dir,
            detector_fn=mock_detector,
        )

        metrics = study["tradeoff_metrics"]
        # In degraded, 102 was missed (total missed = 1)
        # In restored, 102 was recovered (total missed = 0) -> misses_recovered = 1
        assert metrics["misses_recovered"] == 1
        # In restored, background FP was introduced on 101 -> background_fp_introduced = 1
        assert metrics["background_fp_introduced"] == 1

        assert "tradeoff_diagnosis" in study
        summary_file = out_dir / "test_model_low_light_sev3_low_light_enhancement_tradeoff.json"
        assert summary_file.is_file()
