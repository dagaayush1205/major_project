"""Experiment Pipeline and Analysis Orchestrator.

Orchestrates clean vs. degraded object detection experiments across degradation types
and severity levels. Measures aggregate metrics (mAP) alongside systematic error decomposition
to quantify where and why accuracy is lost.
"""

from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
import argparse
import json

from degradations import degrade_directory, AVAILABLE_DEGRADATIONS
from error_analysis import decompose_errors, load_coco_ground_truth
from detect import load_config, run_yolo, run_fasterrcnn, run_rtdetr


MODEL_RUNNERS: Dict[str, Callable[..., None]] = {
    "yolo11": run_yolo,
    "fasterrcnn": run_fasterrcnn,
    "rtdetr": run_rtdetr,
}


def compute_coco_map(gt_source: Union[str, Path], pred_path: Union[str, Path]) -> Optional[Dict[str, float]]:
    """
    Computes standard COCO mAP metrics using pycocotools if available and GT is a valid file.
    Returns None gracefully if pycocotools is unavailable or annotations cannot be loaded.
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        import io
        import contextlib

        gt_path_str = str(gt_source)
        if not Path(gt_path_str).is_file():
            return None

        with open(pred_path, "r") as f:
            predictions = json.load(f)

        if not predictions:
            return {"mAP_50_95": 0.0, "mAP_50": 0.0, "mAP_75": 0.0}

        image_ids = sorted(set(pred["image_id"] for pred in predictions))

        # Suppress standard terminal spam from pycocotools during evaluation
        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt = COCO(gt_path_str)
            coco_dt = coco_gt.loadRes(str(pred_path))
            evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
            evaluator.params.imgIds = image_ids
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()

        stats = evaluator.stats
        return {
            "mAP_50_95": float(stats[0]),
            "mAP_50": float(stats[1]),
            "mAP_75": float(stats[2]),
        }
    except Exception:
        return None


def compare_error_profiles(baseline_summary: Dict[str, Any], condition_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes the diagnostic shift between a baseline (e.g. clean) condition and a degraded condition.
    Identifies the primary failure mode causing accuracy loss.

    Args:
        baseline_summary: Output from error_analysis.decompose_errors on baseline.
        condition_summary: Output from error_analysis.decompose_errors on degraded condition.

    Returns:
        Dictionary containing metric deltas and identified primary failure mode.
    """
    b_sum = baseline_summary["summary"]
    c_sum = condition_summary["summary"]

    delta_miss_rate = round(c_sum["miss_rate"] - b_sum["miss_rate"], 4)
    delta_precision = round(c_sum["precision"] - b_sum["precision"], 4)
    delta_recall = round(c_sum["recall"] - b_sum["recall"], 4)

    b_raw = baseline_summary["raw_counts"]
    c_raw = condition_summary["raw_counts"]

    delta_counts = {
        "delta_tp": c_raw["tp"] - b_raw["tp"],
        "delta_miss": c_raw["miss"] - b_raw["miss"],
        "delta_bkg": c_raw["bkg"] - b_raw["bkg"],
        "delta_loc": c_raw["loc"] - b_raw["loc"],
        "delta_cls": c_raw["cls"] - b_raw["cls"],
        "delta_cls_loc": c_raw["cls_loc"] - b_raw["cls_loc"],
        "delta_dupe": c_raw["dupe"] - b_raw["dupe"],
        "delta_total_det": c_raw["total_det"] - b_raw["total_det"],
    }

    # Identify primary failure mechanism
    # Compare raw increases in error counts
    increases = {
        "Missed Detections (Object Loss)": max(0, delta_counts["delta_miss"]),
        "Background False Positives (Hallucinations)": max(0, delta_counts["delta_bkg"]),
        "Localization Errors (Poor Bounding Boxes)": max(0, delta_counts["delta_loc"]),
        "Classification Errors (Class Confusion)": max(0, delta_counts["delta_cls"]),
        "Class + Localization Errors": max(0, delta_counts["delta_cls_loc"]),
        "Duplicate Detections (NMS Failures)": max(0, delta_counts["delta_dupe"]),
    }

    primary_mode = max(increases.items(), key=lambda item: item[1])
    if primary_mode[1] == 0:
        diagnosis = "No degradation observed or performance identical to baseline."
    else:
        diagnosis = f"Primary Failure Mechanism: {primary_mode[0]} (+{primary_mode[1]} occurrences)"

    return {
        "delta_rates": {
            "delta_miss_rate": delta_miss_rate,
            "delta_precision": delta_precision,
            "delta_recall": delta_recall,
        },
        "delta_counts": delta_counts,
        "diagnosis": diagnosis,
    }


def run_experiment_condition(
    images_dir: Union[str, Path],
    gt_source: Union[str, Path, Dict[str, Any]],
    output_pred_path: Union[str, Path],
    model_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    detector_fn: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
    pos_thresh: float = 0.5,
    loc_thresh: float = 0.1,
    score_thresh: float = 0.0,
) -> Dict[str, Any]:
    """
    Executes detection inference (or uses an injected detector_fn) and runs error decomposition.
    """
    out_pred = Path(output_pred_path)
    out_pred.parent.mkdir(parents=True, exist_ok=True)

    if detector_fn is not None:
        detector_fn(str(images_dir), str(out_pred), config or {})
    elif model_name in MODEL_RUNNERS:
        cfg = config or load_config()
        runner = MODEL_RUNNERS[model_name]
        runner(str(images_dir), str(out_pred), cfg)
    else:
        raise ValueError(
            f"Either detector_fn or a valid model_name ({list(MODEL_RUNNERS.keys())}) must be provided."
        )

    # Decompose errors
    decomp_results = decompose_errors(
        gt_source=gt_source,
        pred_source=str(out_pred),
        pos_thresh=pos_thresh,
        loc_thresh=loc_thresh,
        score_thresh=score_thresh,
    )

    # Calculate mAP if possible
    map_metrics = compute_coco_map(gt_source, out_pred)
    decomp_results["coco_metrics"] = map_metrics

    return decomp_results


def run_degradation_experiment(
    clean_images_dir: Union[str, Path],
    gt_source: Union[str, Path, Dict[str, Any]],
    model_name: str,
    degradation_type: str,
    severities: List[int] = [1, 2, 3, 4, 5],
    output_dir: Union[str, Path] = "results",
    config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    detector_fn: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
    pos_thresh: float = 0.5,
    loc_thresh: float = 0.1,
    score_thresh: float = 0.0,
) -> Dict[str, Any]:
    """
    Runs an end-to-end clean vs. degraded experiment across multiple severities.

    Returns:
        Structured dictionary summarizing clean baseline, per-severity performance,
        error decomposition shifts, and failure mode diagnoses.
    """
    if degradation_type not in AVAILABLE_DEGRADATIONS:
        raise ValueError(f"Unknown degradation '{degradation_type}'. Available: {AVAILABLE_DEGRADATIONS}")

    out_base = Path(output_dir)
    clean_dir = Path(clean_images_dir)
    cfg = config or load_config()

    print(f"\n=======================================================")
    print(f"EXPERIMENT: Model={model_name} | Degradation={degradation_type}")
    print(f"=======================================================\n")

    # Step 1: Clean Baseline
    print(">>> Running Clean Baseline...")
    clean_pred_path = out_base / "predictions" / f"{model_name}_clean.json"
    clean_res = run_experiment_condition(
        images_dir=clean_dir,
        gt_source=gt_source,
        output_pred_path=clean_pred_path,
        model_name=model_name,
        config=cfg,
        detector_fn=detector_fn,
        pos_thresh=pos_thresh,
        loc_thresh=loc_thresh,
        score_thresh=score_thresh,
    )

    clean_summary = clean_res["summary"]
    clean_map = clean_res.get("coco_metrics")
    clean_map_str = f"mAP={clean_map['mAP_50_95']:.3f}" if clean_map else "mAP=N/A"
    print(
        f"Clean Baseline: Recall={clean_summary['recall']:.1%}, "
        f"Miss Rate={clean_summary['miss_rate']:.1%}, "
        f"FPs={clean_summary['total_detections'] - clean_summary['true_positives']} ({clean_map_str})"
    )

    severity_results: Dict[int, Any] = {}
    comparison_table: List[Dict[str, Any]] = []

    # Step 2: Iterate over severities
    for sev in severities:
        print(f"\n>>> Running Severity {sev}...")
        deg_img_dir = out_base / "degraded_images" / f"{degradation_type}_sev{sev}"
        degrade_directory(
            input_dir=clean_dir,
            output_dir=deg_img_dir,
            degradation_type=degradation_type,
            severity=sev,
            seed=seed,
        )

        deg_pred_path = out_base / "predictions" / f"{model_name}_{degradation_type}_sev{sev}.json"
        deg_res = run_experiment_condition(
            images_dir=deg_img_dir,
            gt_source=gt_source,
            output_pred_path=deg_pred_path,
            model_name=model_name,
            config=cfg,
            detector_fn=detector_fn,
            pos_thresh=pos_thresh,
            loc_thresh=loc_thresh,
            score_thresh=score_thresh,
        )

        # Compare against clean baseline
        comp = compare_error_profiles(baseline_summary=clean_res, condition_summary=deg_res)
        deg_res["comparison_with_clean"] = comp

        deg_sum = deg_res["summary"]
        deg_map = deg_res.get("coco_metrics")
        deg_map_val = deg_map["mAP_50_95"] if deg_map else None

        severity_results[sev] = deg_res

        row = {
            "severity": sev,
            "mAP_50_95": deg_map_val,
            "recall": deg_sum["recall"],
            "miss_rate": deg_sum["miss_rate"],
            "delta_miss_rate": comp["delta_rates"]["delta_miss_rate"],
            "background_fp": deg_sum["background_false_positives"],
            "localization_err": deg_sum["localization_errors"],
            "classification_err": deg_sum["classification_errors"],
            "duplicate_det": deg_sum["duplicate_detections"],
            "primary_diagnosis": comp["diagnosis"],
        }
        comparison_table.append(row)

        print(
            f"  Sev {sev}: Miss Rate={deg_sum['miss_rate']:.1%} (Δ={comp['delta_rates']['delta_miss_rate']:+.1%}), "
            f"Bkg FP={deg_sum['background_false_positives']}, "
            f"Loc Err={deg_sum['localization_errors']}"
        )
        print(f"  -> {comp['diagnosis']}")

    full_experiment = {
        "model": model_name,
        "degradation": degradation_type,
        "severities": severities,
        "clean_baseline": clean_res,
        "severities_data": severity_results,
        "comparison_table": comparison_table,
    }

    # Save experiment summary JSON
    summary_path = out_base / f"{model_name}_{degradation_type}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(full_experiment, f, indent=2)

    print(f"\n=======================================================")
    print(f"Experiment complete. Summary saved to: {summary_path}")
    print(f"=======================================================\n")

    return full_experiment


def main():
    parser = argparse.ArgumentParser(description="Run Clean vs. Degraded Object Detection Experiment")
    parser.add_argument("--images", required=True, help="Path to clean images directory")
    parser.add_argument("--gt", required=True, help="Path to COCO ground truth JSON")
    parser.add_argument("--model", required=True, choices=["yolo11", "fasterrcnn", "rtdetr"])
    parser.add_argument("--degradation", required=True, choices=AVAILABLE_DEGRADATIONS)
    parser.add_argument("--severities", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--out-dir", default="results", help="Directory to save predictions and metrics")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration YAML")

    args = parser.parse_args()
    config = load_config(args.config)

    run_degradation_experiment(
        clean_images_dir=args.images,
        gt_source=args.gt,
        model_name=args.model,
        degradation_type=args.degradation,
        severities=args.severities,
        output_dir=args.out_dir,
        config=config,
    )


if __name__ == "__main__":
    main()
