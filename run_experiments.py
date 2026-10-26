"""Unified Master Experiment Driver.

Executes the full research experimental suite:
1. Evaluates multiple detector families (YOLO11, Faster R-CNN, RT-DETR) on clean vs. degraded conditions.
2. Applies 5 calibrated visual degradations across severity levels 1 to 5.
3. Performs systematic diagnostic error decomposition (Miss, Bkg FP, Loc, Cls, Dupe).
4. Conducts image restoration trade-off studies (measuring recovered misses vs. background hallucinations).
5. Automatically generates publication-ready figures (stacked error charts, metric trajectories, model comparisons)
   and thesis summary tables (CSV, Markdown, LaTeX).
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import argparse
import json
import numpy as np
from PIL import Image

from degradations import AVAILABLE_DEGRADATIONS
from detect import load_config
from pipeline import run_degradation_experiment
from restoration import run_restoration_tradeoff_study
from visualize import (
    plot_error_breakdown,
    plot_metric_trajectories,
    plot_model_comparison,
    export_summary_table,
)
from prepare_sample_data import create_synthetic_benchmark


DEFAULT_MODELS = ["yolo11", "fasterrcnn", "rtdetr"]
DEFAULT_DEGRADATIONS = ["low_light", "gaussian_blur", "gaussian_noise", "fog", "jpeg_compression"]

# Pairing degradations with standard restoration methods
RESTORATION_PAIRS = {
    "low_light": "low_light_enhancement",
    "fog": "dehazing",
    "gaussian_noise": "denoising",
}


def make_simulated_detector(gt_source: Union[str, Path, Dict[str, Any]]):
    """
    Creates a simulated detector for fast dry-run / CPU verification.
    Models realistic failure mechanisms under degradation:
    - Dark images -> lower confidence, missed detections.
    - Noisy images -> background hallucinations.
    - Blurred images -> bounding box localization slips.
    """
    if isinstance(gt_source, (str, Path)):
        with open(gt_source, "r") as f:
            gt_data = json.load(f)
    else:
        gt_data = gt_source

    annotations = gt_data.get("annotations", [])
    gt_by_img = {}
    for ann in annotations:
        iid = int(ann["image_id"])
        if iid not in gt_by_img:
            gt_by_img[iid] = []
        gt_by_img[iid].append(ann)

    def simulated_detector(images_dir: str, out_pred_path: str, config: Dict[str, Any]):
        in_path = Path(images_dir)
        folder = in_path.name
        preds = []

        for p in in_path.iterdir():
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                img_id = int(p.stem)
                gts = gt_by_img.get(img_id, [])

                with Image.open(p) as img:
                    arr = np.array(img.convert("RGB"))

                mean_val = float(np.mean(arr))
                std_val = float(np.std(arr))

                # Clean baseline
                if "clean" in folder:
                    for g in gts:
                        preds.append(
                            {
                                "image_id": img_id,
                                "category_id": g["category_id"],
                                "bbox": list(g["bbox"]),
                                "score": 0.95,
                            }
                        )

                # Low light / dark condition: objects below luminance threshold are missed
                elif "low_light" in folder:
                    for g in gts:
                        if mean_val > 60:
                            preds.append(
                                {
                                    "image_id": img_id,
                                    "category_id": g["category_id"],
                                    "bbox": list(g["bbox"]),
                                    "score": 0.85,
                                }
                            )
                        elif mean_val > 30:
                            # Dim: 50% detection probability
                            preds.append(
                                {
                                    "image_id": img_id,
                                    "category_id": g["category_id"],
                                    "bbox": list(g["bbox"]),
                                    "score": 0.55,
                                }
                            )

                # Noise condition: sensor noise triggers false positive hallucinations
                elif "gaussian_noise" in folder:
                    for g in gts:
                        preds.append(
                            {
                                "image_id": img_id,
                                "category_id": g["category_id"],
                                "bbox": list(g["bbox"]),
                                "score": 0.80,
                            }
                        )
                    # Add background hallucination
                    preds.append(
                        {
                            "image_id": img_id,
                            "category_id": 1,
                            "bbox": [5.0, 5.0, 30.0, 30.0],
                            "score": 0.65,
                        }
                    )

                # Blur condition: edge softening causes localization error
                elif "gaussian_blur" in folder:
                    for g in gts:
                        bx, by, bw, bh = g["bbox"]
                        # Shifted box with lower IoU
                        preds.append(
                            {
                                "image_id": img_id,
                                "category_id": g["category_id"],
                                "bbox": [bx + bw * 0.4, by, bw, bh],
                                "score": 0.70,
                            }
                        )

                # Restored condition: recovers missed objects but may introduce hallucination
                elif "restored" in folder:
                    for g in gts:
                        preds.append(
                            {
                                "image_id": img_id,
                                "category_id": g["category_id"],
                                "bbox": list(g["bbox"]),
                                "score": 0.88,
                            }
                        )
                    # Restoration artifact false alarm
                    preds.append(
                        {
                            "image_id": img_id,
                            "category_id": 1,
                            "bbox": [10.0, 10.0, 25.0, 25.0],
                            "score": 0.60,
                        }
                    )

                else:
                    # General degraded fallback
                    for g in gts:
                        preds.append(
                            {
                                "image_id": img_id,
                                "category_id": g["category_id"],
                                "bbox": list(g["bbox"]),
                                "score": 0.75,
                            }
                        )

        Path(out_pred_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_pred_path, "w") as f:
            json.dump(preds, f, indent=2)

    return simulated_detector


def run_full_experiment_suite(
    images_dir: Union[str, Path],
    gt_source: Union[str, Path, Dict[str, Any]],
    models: List[str] = DEFAULT_MODELS,
    degradations: List[str] = DEFAULT_DEGRADATIONS,
    severities: List[int] = [1, 2, 3, 4, 5],
    output_dir: Union[str, Path] = "experiments",
    config_path: str = "config.yaml",
    enable_restoration: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Executes the comprehensive experimental matrix.
    """
    out_base = Path(output_dir)
    figures_dir = out_base / "figures"
    tables_dir = out_base / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)

    detector_fn = make_simulated_detector(gt_source) if dry_run else None

    master_report: Dict[str, Any] = {
        "models": models,
        "degradations": degradations,
        "severities": severities,
        "experiments": {},
        "restoration_studies": {},
    }

    # Iterate over degradations and models
    for deg in degradations:
        summary_paths_for_deg = []

        for model in models:
            exp_out_dir = out_base / f"{model}_{deg}"
            exp_data = run_degradation_experiment(
                clean_images_dir=images_dir,
                gt_source=gt_source,
                model_name=model,
                degradation_type=deg,
                severities=severities,
                output_dir=exp_out_dir,
                config=config,
                detector_fn=detector_fn,
            )

            exp_key = f"{model}_{deg}"
            master_report["experiments"][exp_key] = exp_data["comparison_table"]

            summary_json = exp_out_dir / f"{model}_{deg}_summary.json"
            summary_paths_for_deg.append(summary_json)

            # Auto-generate individual figures and tables
            fig_breakdown = figures_dir / f"{model}_{deg}_error_breakdown.png"
            plot_error_breakdown(exp_data, fig_breakdown)

            fig_traj = figures_dir / f"{model}_{deg}_trajectories.png"
            plot_metric_trajectories(exp_data, fig_traj)

            tbl_prefix = tables_dir / f"{model}_{deg}_table"
            export_summary_table(exp_data, tbl_prefix)

        # Cross-model comparison plot for this degradation
        if len(models) > 1 and len(summary_paths_for_deg) > 1:
            fig_comp = figures_dir / f"model_comparison_{deg}.png"
            plot_model_comparison(summary_paths_for_deg, fig_comp, metric="delta_miss_rate")

    # Optional restoration trade-off studies
    if enable_restoration:
        for deg, rest in RESTORATION_PAIRS.items():
            if deg in degradations:
                # Run trade-off study on severity 3 (moderate degradation)
                study_dir = out_base / "restoration_studies" / f"{deg}_sev3"
                for model in models:
                    study_data = run_restoration_tradeoff_study(
                        clean_images_dir=images_dir,
                        gt_source=gt_source,
                        model_name=model,
                        degradation_type=deg,
                        severity=3,
                        restoration_type=rest,
                        output_dir=study_dir,
                        detector_fn=detector_fn,
                        config=config,
                    )
                    master_report["restoration_studies"][f"{model}_{deg}_sev3_{rest}"] = study_data["tradeoff_metrics"]

    # Save master experiment report
    master_file = out_base / "master_experiment_report.json"
    with open(master_file, "w") as f:
        json.dump(master_report, f, indent=2)

    print(f"\n=======================================================")
    print(f"MASTER SUITE COMPLETE")
    print(f"Reports & Summary : {master_file}")
    print(f"Figures saved to  : {figures_dir}")
    print(f"Tables saved to   : {tables_dir}")
    print(f"=======================================================\n")

    return master_report


def main():
    parser = argparse.ArgumentParser(description="Master Research Experiment Driver")
    parser.add_argument("--images", help="Path to clean images directory")
    parser.add_argument("--gt", help="Path to COCO ground truth annotations JSON")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=DEFAULT_MODELS)
    parser.add_argument("--degradations", nargs="+", default=DEFAULT_DEGRADATIONS, choices=AVAILABLE_DEGRADATIONS)
    parser.add_argument("--severities", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--out-dir", default="experiments", help="Output directory")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration YAML")
    parser.add_argument("--restore", action="store_true", help="Enable image restoration trade-off study")
    parser.add_argument("--dry-run", action="store_true", help="Run simulated inference for fast CPU verification")
    parser.add_argument("--use-synthetic-data", action="store_true", help="Generate synthetic sample data if no dataset provided")

    args = parser.parse_args()

    images_dir = args.images
    gt_path = args.gt

    # Auto-generate synthetic data if requested or if no dataset is provided
    if args.use_synthetic_data or not images_dir or not gt_path:
        print("Dataset not explicitly specified. Generating synthetic benchmark sample...")
        sample_paths = create_synthetic_benchmark(output_dir="sample_benchmark", num_images=4)
        images_dir = str(sample_paths["images_dir"])
        gt_path = str(sample_paths["annotations_path"])
        print(f"Using sample images: {images_dir}")
        print(f"Using sample GT    : {gt_path}")

    run_full_experiment_suite(
        images_dir=images_dir,
        gt_source=gt_path,
        models=args.models,
        degradations=args.degradations,
        severities=args.severities,
        output_dir=args.out_dir,
        config_path=args.config,
        enable_restoration=args.restore,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
