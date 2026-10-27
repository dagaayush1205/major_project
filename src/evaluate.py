import argparse
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gt", required=True, help="Ground-truth COCO annotation JSON")

    parser.add_argument(
        "--pred", required=True, help="Prediction JSON in COCO detection format"
    )

    args = parser.parse_args()

    # Load predictions ourselves first
    with open(args.pred, "r") as f:
        predictions = json.load(f)

    # Get exactly the image IDs for which predictions were generated
    image_ids = sorted(set(pred["image_id"] for pred in predictions))

    print(f"Evaluating on {len(image_ids)} images")

    # Load COCO ground truth
    coco_gt = COCO(args.gt)

    # Load predictions
    coco_dt = coco_gt.loadRes(args.pred)

    # COCO evaluator
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")

    # CRITICAL:
    # evaluate only the same images that were run through the detector
    evaluator.params.imgIds = image_ids

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()


if __name__ == "__main__":
    main()
