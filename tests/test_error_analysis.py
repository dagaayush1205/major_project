import pytest
from error_analysis import (
    box_iou,
    match_image_detections,
    decompose_errors,
    load_coco_ground_truth,
    load_predictions,
)


def test_box_iou_exact():
    # Identical boxes
    box_a = [10.0, 10.0, 20.0, 20.0]
    assert box_iou(box_a, box_a) == pytest.approx(1.0)

    # Disjoint boxes
    box_b = [50.0, 50.0, 20.0, 20.0]
    assert box_iou(box_a, box_b) == 0.0

    # 50% width overlap:
    # box1: [0, 0, 10, 10], area = 100
    # box2: [5, 0, 10, 10], area = 100
    # inter: [5, 0, 5, 10], area = 50
    # union: 100 + 100 - 50 = 150 -> IoU = 50 / 150 = 1/3
    box1 = [0.0, 0.0, 10.0, 10.0]
    box2 = [5.0, 0.0, 10.0, 10.0]
    assert box_iou(box1, box2) == pytest.approx(1.0 / 3.0)


def test_box_iou_degenerate():
    # Negative or zero dimensions
    assert box_iou([0, 0, 0, 10], [0, 0, 10, 10]) == 0.0
    assert box_iou([0, 0, -5, 10], [0, 0, 10, 10]) == 0.0


def test_true_positive_and_miss_detection():
    gt = [{"category_id": 1, "bbox": [10, 10, 50, 50]}]
    det = [{"category_id": 1, "bbox": [10, 10, 50, 50], "score": 0.95}]

    res = match_image_detections(gt, det, pos_thresh=0.5)
    c = res["counts"]
    assert c["tp"] == 1
    assert c["miss"] == 0
    assert c["bkg"] == 0
    assert c["loc"] == 0
    assert c["cls"] == 0
    assert c["dupe"] == 0
    assert c["total_det"] == 1
    assert c["total_gt"] == 1


def test_missed_detection_when_no_prediction():
    gt = [{"category_id": 1, "bbox": [10, 10, 50, 50]}]
    det = []

    res = match_image_detections(gt, det, pos_thresh=0.5)
    c = res["counts"]
    assert c["tp"] == 0
    assert c["miss"] == 1
    assert c["total_det"] == 0
    assert c["total_gt"] == 1


def test_background_false_positive():
    gt = [{"category_id": 1, "bbox": [10, 10, 20, 20]}]
    # Detection far away with IoU = 0
    det = [{"category_id": 1, "bbox": [100, 100, 20, 20], "score": 0.85}]

    res = match_image_detections(gt, det, pos_thresh=0.5, loc_thresh=0.1)
    c = res["counts"]
    assert c["tp"] == 0
    assert c["miss"] == 1
    assert c["bkg"] == 1
    assert c["loc"] == 0
    assert c["cls"] == 0


def test_localization_error():
    # box1: [0, 0, 10, 10]
    # box2: [5, 0, 10, 10] -> IoU = 0.333 (between loc_thresh 0.1 and pos_thresh 0.5)
    gt = [{"category_id": 1, "bbox": [0, 0, 10, 10]}]
    det = [{"category_id": 1, "bbox": [5, 0, 10, 10], "score": 0.9}]

    res = match_image_detections(gt, det, pos_thresh=0.5, loc_thresh=0.1)
    c = res["counts"]
    assert c["tp"] == 0
    assert c["miss"] == 1
    assert c["loc"] == 1
    assert c["bkg"] == 0


def test_classification_error():
    # Perfect box overlap (IoU = 1.0), but different category
    gt = [{"category_id": 1, "bbox": [10, 10, 40, 40]}]
    det = [{"category_id": 2, "bbox": [10, 10, 40, 40], "score": 0.9}]

    res = match_image_detections(gt, det, pos_thresh=0.5, loc_thresh=0.1)
    c = res["counts"]
    assert c["tp"] == 0
    assert c["miss"] == 1
    assert c["cls"] == 1
    assert c["bkg"] == 0
    assert c["loc"] == 0


def test_class_plus_localization_error():
    # Wrong category AND IoU in [0.1, 0.5)
    gt = [{"category_id": 1, "bbox": [0, 0, 10, 10]}]
    det = [{"category_id": 2, "bbox": [5, 0, 10, 10], "score": 0.88}]

    res = match_image_detections(gt, det, pos_thresh=0.5, loc_thresh=0.1)
    c = res["counts"]
    assert c["tp"] == 0
    assert c["miss"] == 1
    assert c["cls_loc"] == 1
    assert c["cls"] == 0
    assert c["loc"] == 0
    assert c["bkg"] == 0


def test_duplicate_detection():
    # Two detections of same class matching same GT with high IoU
    gt = [{"category_id": 1, "bbox": [10, 10, 50, 50]}]
    det = [
        {"category_id": 1, "bbox": [10, 10, 50, 50], "score": 0.95},
        {"category_id": 1, "bbox": [11, 11, 49, 49], "score": 0.85},
    ]

    res = match_image_detections(gt, det, pos_thresh=0.5)
    c = res["counts"]
    assert c["tp"] == 1
    assert c["dupe"] == 1
    assert c["miss"] == 0
    assert c["bkg"] == 0
    assert c["total_det"] == 2
    assert c["total_gt"] == 1


def test_conservation_laws():
    # Complex scenario with multiple GTs and detections
    gt = [
        {"category_id": 1, "bbox": [0, 0, 20, 20]},
        {"category_id": 2, "bbox": [50, 50, 20, 20]},
        {"category_id": 3, "bbox": [100, 100, 20, 20]},
    ]
    det = [
        {"category_id": 1, "bbox": [0, 0, 20, 20], "score": 0.99},  # TP for GT 1
        {"category_id": 1, "bbox": [1, 1, 19, 19], "score": 0.90},  # Dupe for GT 1
        {"category_id": 2, "bbox": [58, 50, 20, 20], "score": 0.85}, # Loc for GT 2 (IoU ~ 0.43)
        {"category_id": 4, "bbox": [100, 100, 20, 20], "score": 0.80}, # Cls for GT 3 (cat 4 != 3)
        {"category_id": 5, "bbox": [200, 200, 20, 20], "score": 0.75}, # Bkg
    ]

    res = match_image_detections(gt, det, pos_thresh=0.5, loc_thresh=0.1)
    c = res["counts"]

    # Exact conservation: Total Dets == TP + Dupe + Loc + Cls + Cls_Loc + Bkg
    assert c["total_det"] == (c["tp"] + c["dupe"] + c["loc"] + c["cls"] + c["cls_loc"] + c["bkg"])
    # Exact conservation: Total GT == TP + Miss
    assert c["total_gt"] == (c["tp"] + c["miss"])

    assert c["tp"] == 1
    assert c["dupe"] == 1
    assert c["loc"] == 1
    assert c["cls"] == 1
    assert c["bkg"] == 1
    assert c["miss"] == 2  # GT 2 and GT 3 had no TP match


def test_dataset_level_decompose_errors():
    gt_dict = {
        "annotations": [
            {"image_id": 101, "category_id": 1, "bbox": [10, 10, 30, 30], "id": 1},
            {"image_id": 102, "category_id": 2, "bbox": [20, 20, 40, 40], "id": 2},
        ]
    }
    preds_list = [
        {"image_id": 101, "category_id": 1, "bbox": [10, 10, 30, 30], "score": 0.95},
        {"image_id": 102, "category_id": 2, "bbox": [200, 200, 40, 40], "score": 0.90},
    ]

    results = decompose_errors(gt_dict, preds_list, pos_thresh=0.5)
    s = results["summary"]
    assert s["total_ground_truth"] == 2
    assert s["total_detections"] == 2
    assert s["true_positives"] == 1
    assert s["missed_detections"] == 1
    assert s["background_false_positives"] == 1
    assert s["recall"] == 0.5
    assert s["miss_rate"] == 0.5
