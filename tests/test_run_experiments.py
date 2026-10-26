import tempfile
from pathlib import Path
import json
import pytest

from prepare_sample_data import create_synthetic_benchmark
from run_experiments import run_full_experiment_suite, make_simulated_detector


def test_create_synthetic_benchmark():
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_paths = create_synthetic_benchmark(output_dir=tmpdir, num_images=3, seed=123)

        images_dir = sample_paths["images_dir"]
        annotations_path = sample_paths["annotations_path"]

        assert images_dir.is_dir()
        assert annotations_path.is_file()

        # Check generated image files
        jpg_files = list(images_dir.glob("*.jpg"))
        assert len(jpg_files) == 3

        # Check annotation JSON schema
        with open(annotations_path, "r") as f:
            data = json.load(f)

        assert "images" in data and len(data["images"]) == 3
        assert "annotations" in data and len(data["annotations"]) >= 3
        assert "categories" in data and len(data["categories"]) > 0

        first_ann = data["annotations"][0]
        assert "bbox" in first_ann and len(first_ann["bbox"]) == 4
        assert first_ann["bbox"][2] > 0 and first_ann["bbox"][3] > 0


def test_simulated_detector():
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_paths = create_synthetic_benchmark(output_dir=tmpdir, num_images=2)
        images_dir = sample_paths["images_dir"]
        annotations_path = sample_paths["annotations_path"]

        detector_fn = make_simulated_detector(annotations_path)
        out_pred = Path(tmpdir) / "preds.json"

        detector_fn(str(images_dir), str(out_pred), {})
        assert out_pred.is_file()

        with open(out_pred, "r") as f:
            preds = json.load(f)

        assert isinstance(preds, list)
        assert len(preds) > 0
        assert "bbox" in preds[0]
        assert "score" in preds[0]
        assert "category_id" in preds[0]


def test_run_full_experiment_suite_dry_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        sample_dir = tmp_path / "sample"
        exp_dir = tmp_path / "experiments"

        # Generate sample benchmark
        sample_paths = create_synthetic_benchmark(output_dir=sample_dir, num_images=2, seed=42)

        # Run master suite in dry-run mode on 2 models and 2 degradations with restoration
        report = run_full_experiment_suite(
            images_dir=sample_paths["images_dir"],
            gt_source=sample_paths["annotations_path"],
            models=["yolo11", "fasterrcnn"],
            degradations=["low_light", "gaussian_noise"],
            severities=[1, 2],
            output_dir=exp_dir,
            enable_restoration=True,
            dry_run=True,
        )

        assert "experiments" in report
        assert "yolo11_low_light" in report["experiments"]
        assert "fasterrcnn_gaussian_noise" in report["experiments"]
        assert "restoration_studies" in report

        # Verify output files
        master_json = exp_dir / "master_experiment_report.json"
        assert master_json.is_file()

        # Verify figures
        fig_dir = exp_dir / "figures"
        assert (fig_dir / "yolo11_low_light_error_breakdown.png").is_file()
        assert (fig_dir / "yolo11_low_light_trajectories.png").is_file()
        assert (fig_dir / "model_comparison_low_light.png").is_file()

        # Verify tables
        tbl_dir = exp_dir / "tables"
        assert (tbl_dir / "yolo11_low_light_table.csv").is_file()
        assert (tbl_dir / "yolo11_low_light_table.md").is_file()
        assert (tbl_dir / "yolo11_low_light_table.tex").is_file()
