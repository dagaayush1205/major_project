import yaml
import argparse
import json
from pathlib import Path

from PIL import Image


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# YOLO class index -> official COCO category ID
COCO80_TO_COCO91 = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
]


def get_image_id(path):
    """
    For COCO images such as:
        000000397133.jpg

    returns:
        397133

    For arbitrary test images, use the filename stem if it is numeric.
    """
    stem = Path(path).stem

    try:
        return int(stem)
    except ValueError:
        raise ValueError(
            f"Image filename '{Path(path).name}' does not contain a numeric "
            "COCO image ID. For now rename test images using numeric filenames."
        )


def run_yolo(images_dir, output_file, config):
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(
            "ultralytics is required to run YOLO11. Install it via 'pip install ultralytics'."
        ) from e

    conf = config["inference"]["confidence_threshold"]
    max_det = config["inference"]["max_detections"]
    model_cfg = config["models"]["yolo11"]

    model = YOLO(model_cfg["checkpoint"])

    results = model.predict(
        source=images_dir,
        imgsz=model_cfg.get("input_size", 640),
        conf=conf,
        max_det=max_det,
        verbose=True,
    )

    coco_results = []

    for result in results:
        image_id = get_image_id(result.path)

        if result.boxes is None:
            continue

        boxes = result.boxes

        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for box, score, cls_idx in zip(xyxy, scores, classes):
            x1, y1, x2, y2 = box

            width = x2 - x1
            height = y2 - y1

            coco_results.append(
                {
                    "image_id": int(image_id),
                    "category_id": COCO80_TO_COCO91[cls_idx],
                    "bbox": [float(x1), float(y1), float(width), float(height)],
                    "score": float(score),
                }
            )

    with open(output_file, "w") as f:
        json.dump(coco_results, f, indent=2)

    print()
    print(f"Saved {len(coco_results)} detections to {output_file}")


def run_fasterrcnn(images_dir, output_file, config):
    try:
        import torch
        from torchvision.transforms.functional import pil_to_tensor
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn,
            FasterRCNN_ResNet50_FPN_Weights,
        )
    except ImportError as e:
        raise ImportError(
            "torch and torchvision are required to run Faster R-CNN. Install them via 'pip install torch torchvision'."
        ) from e

    conf = config["inference"]["confidence_threshold"]
    max_det = config["inference"]["max_detections"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        box_score_thresh=conf,
        box_detections_per_img=max_det,
    )

    model.to(device)
    model.eval()

    image_paths = sorted(
        [
            p
            for p in Path(images_dir).iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]
    )

    coco_results = []

    with torch.no_grad():
        for image_path in image_paths:
            image_id = get_image_id(image_path)

            image = Image.open(image_path).convert("RGB")

            # Convert uint8 PIL image -> float tensor [0,1]
            image_tensor = pil_to_tensor(image).float() / 255.0

            image_tensor = image_tensor.to(device)

            output = model([image_tensor])[0]

            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()

            detection_count = 0

            for box, score, label in zip(boxes, scores, labels):
                # Defensive check against threshold
                if score < conf:
                    continue

                x1, y1, x2, y2 = box

                width = x2 - x1
                height = y2 - y1

                coco_results.append(
                    {
                        "image_id": int(image_id),
                        # torchvision's pretrained COCO labels
                        # already correspond to COCO category IDs
                        "category_id": int(label),
                        "bbox": [float(x1), float(y1), float(width), float(height)],
                        "score": float(score),
                    }
                )

                detection_count += 1

            print(f"{image_path.name}: {detection_count} detections")

    with open(output_file, "w") as f:
        json.dump(coco_results, f, indent=2)

    print()
    print(f"Saved {len(coco_results)} detections to {output_file}")


def run_rtdetr(images_dir, output_file, config):
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(
            "ultralytics is required to run RT-DETR. Install it via 'pip install ultralytics'."
        ) from e

    conf = config["inference"]["confidence_threshold"]
    max_det = config["inference"]["max_detections"]
    model_cfg = config["models"]["rtdetr"]

    model = YOLO(model_cfg["checkpoint"])

    results = model.predict(
        source=images_dir,
        imgsz=model_cfg.get("input_size", 640),
        conf=conf,
        max_det=max_det,
        verbose=True,
    )

    coco_results = []

    for result in results:
        image_id = get_image_id(result.path)

        if result.boxes is None:
            continue

        boxes = result.boxes

        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for box, score, cls_idx in zip(xyxy, scores, classes):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            coco_results.append(
                {
                    "image_id": int(image_id),
                    "category_id": COCO80_TO_COCO91[cls_idx],
                    "bbox": [float(x1), float(y1), float(width), float(height)],
                    "score": float(score),
                }
            )

    with open(output_file, "w") as f:
        json.dump(coco_results, f, indent=2)

    print()
    print(f"Saved {len(coco_results)} detections to {output_file}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", required=True, choices=["yolo11", "fasterrcnn", "rtdetr"]
    )

    parser.add_argument("--images", required=True)

    parser.add_argument("--out", required=True)

    parser.add_argument(
        "--config", default="config.yaml", help="Path to configuration YAML"
    )

    args = parser.parse_args()
    config = load_config(args.config)

    if args.model == "yolo11":
        run_yolo(args.images, args.out, config)

    elif args.model == "fasterrcnn":
        run_fasterrcnn(args.images, args.out, config)

    elif args.model == "rtdetr":
        run_rtdetr(args.images, args.out, config)


if __name__ == "__main__":
    main()
