"""Error Decomposition Module for Object Detection Robustness Research.

Decomposes object detection errors into standard, interpretable diagnostic categories:
- True Positive (TP): Correct class, IoU >= pos_thresh (primary match).
- Missed Detection (Miss / False Negative): Ground truth with no valid TP detection.
- Background Error (Bkg / Hallucination): Detection with IoU < loc_thresh with any GT.
- Localization Error (Loc): Correct class, loc_thresh <= IoU < pos_thresh.
- Classification Error (Cls): Wrong class, IoU >= pos_thresh with another GT.
- Class + Localization Error (Cls_Loc): Wrong class, loc_thresh <= IoU < pos_thresh.
- Duplicate Detection (Dupe): Correct class and IoU >= pos_thresh, but GT was already claimed.

Guarantees exact conservation laws:
  Total Detections = TP + Dupe + Loc + Cls + Cls_Loc + Bkg
  Total Ground Truth = TP + Miss
"""

from typing import Dict, List, Tuple, Any, Optional, Union
import argparse
import json
from pathlib import Path


def box_iou(box1: List[float], box2: List[float]) -> float:
    """
    Computes Intersection-over-Union (IoU) between two bounding boxes in COCO format:
    [x, y, width, height].

    Args:
        box1: [x, y, w, h]
        box2: [x, y, w, h]

    Returns:
        float IoU value in [0.0, 1.0].
    """
    x1_min, y1_min, w1, h1 = box1
    x2_min, y2_min, w2, h2 = box2

    if w1 <= 0.0 or h1 <= 0.0 or w2 <= 0.0 or h2 <= 0.0:
        return 0.0

    x1_max = x1_min + w1
    y1_max = y1_min + h1
    x2_max = x2_min + w2
    y2_max = y2_min + h2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0

    return float(inter_area / union_area)


def match_image_detections(
    ground_truths: List[Dict[str, Any]],
    detections: List[Dict[str, Any]],
    pos_thresh: float = 0.5,
    loc_thresh: float = 0.1,
    score_thresh: float = 0.0,
) -> Dict[str, Any]:
    """
    Categorizes all detections and ground truth objects for a single image.

    Args:
        ground_truths: List of GT dicts with 'category_id' and 'bbox' ([x, y, w, h]).
        detections: List of detection dicts with 'category_id', 'bbox', and 'score'.
        pos_thresh: Minimum IoU for a positive match (default 0.5).
        loc_thresh: Minimum IoU to be considered a localization match rather than background (default 0.1).
        score_thresh: Minimum confidence score to consider (default 0.0).

    Returns:
        Dict containing categorized lists of detections and counts.
    """
    # Filter detections by minimum score and sort descending
    filtered_dets = [
        d for d in detections if float(d.get("score", 1.0)) >= score_thresh
    ]
    sorted_dets = sorted(
        filtered_dets, key=lambda d: float(d.get("score", 0.0)), reverse=True
    )

    # Track which GT objects have been matched by a primary True Positive
    gt_matched = [False] * len(ground_truths)

    tp_dets = []
    dupe_dets = []
    loc_dets = []
    cls_dets = []
    cls_loc_dets = []
    bkg_dets = []

    for det in sorted_dets:
        det_cat = det["category_id"]
        det_box = det["bbox"]

        # Step 1: Check matches against same-category GTs
        best_same_iou = -1.0
        best_same_idx = -1

        for idx, gt in enumerate(ground_truths):
            if gt["category_id"] == det_cat:
                iou = box_iou(det_box, gt["bbox"])
                if iou > best_same_iou:
                    best_same_iou = iou
                    best_same_idx = idx

        # If it has a high-quality overlap with a same-category GT
        if best_same_iou >= pos_thresh:
            if not gt_matched[best_same_idx]:
                # First detection to claim this GT is True Positive
                gt_matched[best_same_idx] = True
                tp_dets.append(det)
                continue
            else:
                # Duplicate detection of an already matched GT
                dupe_dets.append(det)
                continue

        # Step 2: Check for same-category localization error
        if best_same_iou >= loc_thresh:
            loc_dets.append(det)
            continue

        # Step 3: Check matches against different-category GTs
        best_diff_iou = -1.0
        for gt in ground_truths:
            if gt["category_id"] != det_cat:
                iou = box_iou(det_box, gt["bbox"])
                if iou > best_diff_iou:
                    best_diff_iou = iou

        if best_diff_iou >= pos_thresh:
            # Overlaps well with an object of another class -> Classification error
            cls_dets.append(det)
        elif best_diff_iou >= loc_thresh:
            # Moderate overlap with another class -> Class + Loc error
            cls_loc_dets.append(det)
        else:
            # No significant overlap with any GT -> Background false alarm
            bkg_dets.append(det)

    # Any GT not matched by a TP is a Missed Detection
    missed_gts = [
        gt for idx, gt in enumerate(ground_truths) if not gt_matched[idx]
    ]

    return {
        "counts": {
            "tp": len(tp_dets),
            "miss": len(missed_gts),
            "bkg": len(bkg_dets),
            "loc": len(loc_dets),
            "cls": len(cls_dets),
            "cls_loc": len(cls_loc_dets),
            "dupe": len(dupe_dets),
            "total_det": len(sorted_dets),
            "total_gt": len(ground_truths),
        },
        "tp": tp_dets,
        "miss": missed_gts,
        "bkg": bkg_dets,
        "loc": loc_dets,
        "cls": cls_dets,
        "cls_loc": cls_loc_dets,
        "dupe": dupe_dets,
    }


def load_coco_ground_truth(gt_source: Union[str, Path, Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """
    Loads COCO ground truth annotations and groups them by image_id.
    Accepts either a JSON file path or a pre-loaded dictionary.
    """
    if isinstance(gt_source, (str, Path)):
        with open(gt_source, "r") as f:
            data = json.load(f)
    elif isinstance(gt_source, dict):
        data = gt_source
    else:
        raise TypeError(f"Expected path or dict for GT, got {type(gt_source).__name__}")

    annotations = data.get("annotations", [])
    gt_by_image: Dict[int, List[Dict[str, Any]]] = {}

    for ann in annotations:
        # Ignore crowd annotations if specified
        if ann.get("iscrowd", 0) == 1:
            continue
        img_id = int(ann["image_id"])
        if img_id not in gt_by_image:
            gt_by_image[img_id] = []
        gt_by_image[img_id].append(
            {
                "image_id": img_id,
                "category_id": int(ann["category_id"]),
                "bbox": [float(c) for c in ann["bbox"]],
                "id": ann.get("id"),
            }
        )

    return gt_by_image


def load_predictions(pred_source: Union[str, Path, List[Dict[str, Any]]]) -> Dict[int, List[Dict[str, Any]]]:
    """
    Loads detection predictions and groups them by image_id.
    Accepts either a JSON file path or a pre-loaded list of dicts.
    """
    if isinstance(pred_source, (str, Path)):
        with open(pred_source, "r") as f:
            predictions = json.load(f)
    elif isinstance(pred_source, list):
        predictions = pred_source
    else:
        raise TypeError(f"Expected path or list for predictions, got {type(pred_source).__name__}")

    preds_by_image: Dict[int, List[Dict[str, Any]]] = {}

    for pred in predictions:
        img_id = int(pred["image_id"])
        if img_id not in preds_by_image:
            preds_by_image[img_id] = []
        preds_by_image[img_id].append(
            {
                "image_id": img_id,
                "category_id": int(pred["category_id"]),
                "bbox": [float(c) for c in pred["bbox"]],
                "score": float(pred.get("score", 1.0)),
            }
        )

    return preds_by_image


def decompose_errors(
    gt_source: Union[str, Path, Dict[str, Any]],
    pred_source: Union[str, Path, List[Dict[str, Any]]],
    pos_thresh: float = 0.5,
    loc_thresh: float = 0.1,
    score_thresh: float = 0.0,
    image_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Performs full dataset-level error decomposition.

    Args:
        gt_source: Ground truth COCO JSON path or dictionary.
        pred_source: Predictions JSON path or list of detection dicts.
        pos_thresh: Primary IoU threshold (default 0.5).
        loc_thresh: Minimum IoU for localization error (default 0.1).
        score_thresh: Minimum prediction confidence score (default 0.0).
        image_ids: Optional explicit subset of image IDs to evaluate.

    Returns:
        Structured dictionary containing aggregated counts, error rates,
        fractions, and per-image details.
    """
    gt_by_image = load_coco_ground_truth(gt_source)
    preds_by_image = load_predictions(pred_source)

    if image_ids is None:
        target_ids = sorted(set(gt_by_image.keys()) | set(preds_by_image.keys()))
    else:
        target_ids = sorted(list(image_ids))

    agg_counts = {
        "tp": 0,
        "miss": 0,
        "bkg": 0,
        "loc": 0,
        "cls": 0,
        "cls_loc": 0,
        "dupe": 0,
        "total_det": 0,
        "total_gt": 0,
    }

    per_image_results = {}

    for img_id in target_ids:
        img_gts = gt_by_image.get(img_id, [])
        img_dets = preds_by_image.get(img_id, [])

        img_res = match_image_detections(
            ground_truths=img_gts,
            detections=img_dets,
            pos_thresh=pos_thresh,
            loc_thresh=loc_thresh,
            score_thresh=score_thresh,
        )

        per_image_results[img_id] = img_res["counts"]
        for key in agg_counts:
            agg_counts[key] += img_res["counts"][key]

    total_gt = agg_counts["total_gt"]
    total_det = agg_counts["total_det"]
    total_tp = agg_counts["tp"]
    total_fp = total_det - total_tp

    # Rates and diagnostic ratios
    miss_rate = (agg_counts["miss"] / total_gt) if total_gt > 0 else 0.0
    precision = (total_tp / total_det) if total_det > 0 else 0.0
    recall = (total_tp / total_gt) if total_gt > 0 else 0.0

    # False Positive breakdown (% of total false alarms)
    fp_breakdown = {
        "bkg_frac": (agg_counts["bkg"] / total_fp) if total_fp > 0 else 0.0,
        "loc_frac": (agg_counts["loc"] / total_fp) if total_fp > 0 else 0.0,
        "cls_frac": (agg_counts["cls"] / total_fp) if total_fp > 0 else 0.0,
        "cls_loc_frac": (agg_counts["cls_loc"] / total_fp) if total_fp > 0 else 0.0,
        "dupe_frac": (agg_counts["dupe"] / total_fp) if total_fp > 0 else 0.0,
    }

    return {
        "parameters": {
            "pos_thresh": pos_thresh,
            "loc_thresh": loc_thresh,
            "score_thresh": score_thresh,
            "num_images_evaluated": len(target_ids),
        },
        "summary": {
            "total_ground_truth": total_gt,
            "total_detections": total_det,
            "true_positives": total_tp,
            "missed_detections": agg_counts["miss"],
            "background_false_positives": agg_counts["bkg"],
            "localization_errors": agg_counts["loc"],
            "classification_errors": agg_counts["cls"],
            "cls_loc_errors": agg_counts["cls_loc"],
            "duplicate_detections": agg_counts["dupe"],
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "miss_rate": round(miss_rate, 4),
        },
        "raw_counts": agg_counts,
        "fp_breakdown_fractions": {
            k: round(v, 4) for k, v in fp_breakdown.items()
        },
        "per_image": per_image_results,
    }


def print_error_report(results: Dict[str, Any]) -> None:
    """Prints a structured ASCII report of error decomposition results."""
    s = results["summary"]
    p = results["parameters"]
    fp = results["fp_breakdown_fractions"]

    print("=" * 60)
    print("OBJECT DETECTION ERROR DECOMPOSITION REPORT")
    print("=" * 60)
    print(f"Evaluated Images     : {p['num_images_evaluated']}")
    print(f"Thresholds           : IoU_pos={p['pos_thresh']}, IoU_loc={p['loc_thresh']}, score_thresh={p['score_thresh']}")
    print("-" * 60)
    print(f"Ground Truth Objects : {s['total_ground_truth']}")
    print(f"Total Detections     : {s['total_detections']}")
    print(f"True Positives (TP)  : {s['true_positives']}  (Recall: {s['recall']:.2%})")
    print(f"Missed Objects (FN)  : {s['missed_detections']}  (Miss Rate: {s['miss_rate']:.2%})")
    print("-" * 60)
    print("False Positive Breakdown:")
    total_fp = s["total_detections"] - s["true_positives"]
    print(f"  Total False Positives : {total_fp}")
    print(f"  - Background (Hallucinations) : {s['background_false_positives']:>5} ({fp['bkg_frac']:.1%})")
    print(f"  - Localization Errors         : {s['localization_errors']:>5} ({fp['loc_frac']:.1%})")
    print(f"  - Classification Errors       : {s['classification_errors']:>5} ({fp['cls_frac']:.1%})")
    print(f"  - Class + Loc Errors          : {s['cls_loc_errors']:>5} ({fp['cls_loc_frac']:.1%})")
    print(f"  - Duplicate Detections        : {s['duplicate_detections']:>5} ({fp['dupe_frac']:.1%})")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Perform error decomposition on object detector predictions."
    )
    parser.add_argument("--gt", required=True, help="Path to ground-truth COCO annotations JSON")
    parser.add_argument("--pred", required=True, help="Path to detector predictions JSON")
    parser.add_argument("--out", default=None, help="Optional output JSON path for detailed metrics")
    parser.add_argument("--pos-thresh", type=float, default=0.5, help="Positive IoU threshold (default 0.5)")
    parser.add_argument("--loc-thresh", type=float, default=0.1, help="Localization error threshold (default 0.1)")
    parser.add_argument("--score-thresh", type=float, default=0.0, help="Confidence score threshold")

    args = parser.parse_args()

    results = decompose_errors(
        gt_source=args.gt,
        pred_source=args.pred,
        pos_thresh=args.pos_thresh,
        loc_thresh=args.loc_thresh,
        score_thresh=args.score_thresh,
    )

    print_error_report(results)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved full error decomposition results to {args.out}")


if __name__ == "__main__":
    main()
