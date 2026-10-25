import tempfile
from pathlib import Path
import json
import numpy as np
import pytest
from PIL import Image

from pipeline import compare_error_profiles, run_degradation_experiment


def test_compare_error_profiles_diagnoses_object_loss():
    baseline = {
        "summary": {"miss_rate": 0.10, "precision": 0.90, "recall": 0.90},
        "raw_counts": {
            "tp": 90,
            "miss": 10,
            "bkg": 5,
            "loc": 3,
            "cls": 2,
            "cls_loc": 0,
            "dupe": 0,
            "total_det": 100,
            "total_gt": 100,
        },
    }

    # Condition with major drop in detections / increase in missed objects
    degraded = {
        "summary": {"miss_rate": 0.65, "precision": 0.85, "recall": 0.35},
        "raw_counts": {
            "tp": 35,
            "miss": 65,  # +55 misses!
            "bkg": 4,
            "loc": 1,
            "cls": 1,
            "cls_loc": 0,
            "dupe": 0,
            "total_det": 41,
            "total_gt": 100,
        },
    }

    comp = compare_error_profiles(baseline, degraded)
    assert comp["delta_rates"]["delta_miss_rate"] == 0.55
    assert comp["delta_counts"]["delta_miss"] == 55
    assert "Missed Detections (Object Loss)" in comp["diagnosis"]


def test_compare_error_profiles_diagnoses_hallucinations():
    baseline = {
        "summary": {"miss_rate": 0.10, "precision": 0.90, "recall": 0.90},
        "raw_counts": {
            "tp": 90,
            "miss": 10,
            "bkg": 5,
            "loc": 3,
            "cls": 2,
            "cls_loc": 0,
            "dupe": 0,
            "total_det": 100,
            "total_gt": 100,
        },
    }

    # Condition where detector starts hallucinating background noise
    degraded = {
        "summary": {"miss_rate": 0.12, "precision": 0.50, "recall": 0.88},
        "raw_counts": {
            "tp": 88,
            "miss": 12,
            "bkg": 80,  # +75 background false positives!
            "loc": 5,
            "cls": 3,
            "cls_loc": 0,
            "dupe": 0,
            "total_det": 176,
            "total_gt": 100,
        },
    }

    comp = compare_error_profiles(baseline, degraded)
    assert comp["delta_counts"]["delta_bkg"] == 75
    assert "Background False Positives (Hallucinations)" in comp["diagnosis"]


def test_compare_error_profiles_identical():
    baseline = {
        "summary": {"miss_rate": 0.10, "precision": 0.90, "recall": 0.90},
        "raw_counts": {
            "tp": 90,
            "miss": 10,
            "bkg": 5,
            "loc": 3,
            "cls": 2,
            "cls_loc": 0,
            "dupe": 0,
            "total_det": 100,
            "total_gt": 100,
        },
    }

    comp = compare_error_profiles(baseline, baseline)
    assert comp["delta_rates"]["delta_miss_rate"] == 0.0
    assert "identical" in comp["diagnosis"].lower()


def test_run_degradation_experiment_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        clean_dir = tmp_path / "clean_images"
        clean_dir.mkdir()
        out_dir = tmp_path / "results"

        # Create two sample clean images
        img1 = Image.fromarray(np.full((64, 64, 3), 150, dtype=np.uint8), mode="RGB")
        img2 = Image.fromarray(np.full((64, 64, 3), 170, dtype=np.uint8), mode="RGB")
        img1.save(clean_dir / "101.jpg")
        img2.save(clean_dir / "102.jpg")

        # Ground truth COCO format
        gt_dict = {
            "annotations": [
                {"image_id": 101, "category_id": 1, "bbox": [10, 10, 20, 20], "id": 1},
                {"image_id": 102, "category_id": 1, "bbox": [15, 15, 25, 25], "id": 2},
            ]
        }
        gt_path = tmp_path / "gt.json"
        with open(gt_path, "w") as f:
            json.dump(gt_dict, f)

        # Mock detector that simulates lower recall as images get darker
        def mock_detector(images_path, pred_path, config):
            in_p = Path(images_path)
            preds = []
            for img_p in in_p.iterdir():
                if img_p.suffix == ".jpg":
                    img_id = int(img_p.stem)
                    with Image.open(img_p) as im:
                        mean_val = float(np.mean(im))
                    # If bright enough, detect the object
                    if mean_val > 100:
                        preds.append(
                            {
                                "image_id": img_id,
                                "category_id": 1,
                                "bbox": [10.0, 10.0, 20.0, 20.0],
                                "score": 0.95,
                            }
                        )
                    elif mean_val > 50:
                        # Localization slip
                        preds.append(
                            {
                                "image_id": img_id,
                                "category_id": 1,
                                "bbox": [16.0, 10.0, 20.0, 20.0],
                                "score": 0.60,
                            }
                        )
                    # If very dark, misses object entirely
            with open(pred_path, "w") as f:
                json.dump(preds, f)

        # Run experiment with severities 1 and 3
        experiment_results = run_degradation_experiment(
            clean_images_dir=clean_dir,
            gt_source=gt_dict,
            model_name="mock_model",
            degradation_type="low_light",
            severities=[1, 3],
            output_dir=out_dir,
            detector_fn=mock_detector,
        )

        assert experiment_results["model"] == "mock_model"
        assert experiment_results["degradation"] == "low_light"
        assert len(experiment_results["comparison_table"]) == 2

        # Verify artifacts were generated
        summary_file = out_dir / "mock_model_low_light_summary.json"
        assert summary_file.is_file()

        clean_pred = out_dir / "predictions" / "mock_model_clean.json"
        assert clean_pred.is_file()

        deg_sev1_img = out_dir / "degraded_images" / "low_light_sev1" / "101.jpg"
        deg_sev3_img = out_dir / "degraded_images" / "low_light_sev3" / "101.jpg"
        assert deg_sev1_img.is_file()
        assert deg_sev3_img.is_file()
