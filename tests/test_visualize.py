import tempfile
from pathlib import Path
import json
import pytest

from visualize import (
    plot_error_breakdown,
    plot_metric_trajectories,
    plot_model_comparison,
    export_summary_table,
    load_experiment_summary,
)


@pytest.fixture
def mock_experiment_data():
    return {
        "model": "yolo11n",
        "degradation": "low_light",
        "severities": [1, 2, 3],
        "clean_baseline": {
            "summary": {"recall": 0.90, "miss_rate": 0.10, "precision": 0.85},
            "raw_counts": {"tp": 90, "miss": 10, "bkg": 15, "loc": 5, "cls": 2, "cls_loc": 1, "dupe": 2},
            "coco_metrics": {"mAP_50_95": 0.52, "mAP_50": 0.75, "mAP_75": 0.55},
        },
        "severities_data": {
            "1": {
                "summary": {"recall": 0.82, "miss_rate": 0.18, "precision": 0.83},
                "raw_counts": {"tp": 82, "miss": 18, "bkg": 16, "loc": 6, "cls": 3, "cls_loc": 1, "dupe": 1},
                "coco_metrics": {"mAP_50_95": 0.48, "mAP_50": 0.70, "mAP_75": 0.50},
            },
            "2": {
                "summary": {"recall": 0.65, "miss_rate": 0.35, "precision": 0.79},
                "raw_counts": {"tp": 65, "miss": 35, "bkg": 17, "loc": 8, "cls": 4, "cls_loc": 2, "dupe": 1},
                "coco_metrics": {"mAP_50_95": 0.38, "mAP_50": 0.58, "mAP_75": 0.40},
            },
            "3": {
                "summary": {"recall": 0.40, "miss_rate": 0.60, "precision": 0.70},
                "raw_counts": {"tp": 40, "miss": 60, "bkg": 17, "loc": 10, "cls": 5, "cls_loc": 3, "dupe": 0},
                "coco_metrics": {"mAP_50_95": 0.22, "mAP_50": 0.35, "mAP_75": 0.20},
            },
        },
        "comparison_table": [
            {
                "severity": 1,
                "mAP_50_95": 0.48,
                "recall": 0.82,
                "miss_rate": 0.18,
                "delta_miss_rate": 0.08,
                "background_fp": 16,
                "localization_err": 6,
                "classification_err": 3,
                "duplicate_det": 1,
                "primary_diagnosis": "Primary Failure Mechanism: Missed Detections (Object Loss) (+8 occurrences)",
            },
            {
                "severity": 2,
                "mAP_50_95": 0.38,
                "recall": 0.65,
                "miss_rate": 0.35,
                "delta_miss_rate": 0.25,
                "background_fp": 17,
                "localization_err": 8,
                "classification_err": 4,
                "duplicate_det": 1,
                "primary_diagnosis": "Primary Failure Mechanism: Missed Detections (Object Loss) (+25 occurrences)",
            },
            {
                "severity": 3,
                "mAP_50_95": 0.22,
                "recall": 0.40,
                "miss_rate": 0.60,
                "delta_miss_rate": 0.50,
                "background_fp": 17,
                "localization_err": 10,
                "classification_err": 5,
                "duplicate_det": 0,
                "primary_diagnosis": "Primary Failure Mechanism: Missed Detections (Object Loss) (+50 occurrences)",
            },
        ],
    }


def test_plot_error_breakdown(mock_experiment_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        out_png = Path(tmpdir) / "breakdown.png"
        res = plot_error_breakdown(mock_experiment_data, out_png, as_percentage=True)
        assert res.is_file()
        assert res.stat().st_size > 1000

        # Also test absolute count mode
        out_counts = Path(tmpdir) / "counts.png"
        res_counts = plot_error_breakdown(mock_experiment_data, out_counts, as_percentage=False)
        assert res_counts.is_file()
        assert res_counts.stat().st_size > 1000


def test_plot_metric_trajectories(mock_experiment_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        out_png = Path(tmpdir) / "trajectories.png"
        res = plot_metric_trajectories(mock_experiment_data, out_png)
        assert res.is_file()
        assert res.stat().st_size > 1000


def test_plot_model_comparison(mock_experiment_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Save two experiment summaries with different models
        f1 = tmp_path / "model1.json"
        f2 = tmp_path / "model2.json"

        data2 = json.loads(json.dumps(mock_experiment_data))
        data2["model"] = "fasterrcnn"
        for row in data2["comparison_table"]:
            row["delta_miss_rate"] = row["delta_miss_rate"] * 0.8  # slightly different

        with open(f1, "w") as f:
            json.dump(mock_experiment_data, f)
        with open(f2, "w") as f:
            json.dump(data2, f)

        out_png = tmp_path / "comparison.png"
        res = plot_model_comparison([f1, f2], out_png, metric="delta_miss_rate")
        assert res.is_file()
        assert res.stat().st_size > 1000


def test_export_summary_table(mock_experiment_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = Path(tmpdir) / "table_output"
        tables = export_summary_table(mock_experiment_data, prefix)

        assert "csv" in tables and tables["csv"].is_file()
        assert "md" in tables and tables["md"].is_file()
        assert "tex" in tables and tables["tex"].is_file()

        # Check CSV content
        csv_text = tables["csv"].read_text()
        assert "Severity" in csv_text
        assert "Δ Miss Rate" in csv_text

        # Check Markdown content
        md_text = tables["md"].read_text()
        assert "|" in md_text
        assert "Primary Failure Mechanism" in md_text

        # Check LaTeX content
        tex_text = tables["tex"].read_text()
        assert "tabular" in tex_text
