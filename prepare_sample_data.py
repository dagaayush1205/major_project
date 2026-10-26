"""Sample Data Preparation Utility.

Generates reproducible sample datasets and annotations for testing, demonstration,
and benchmark verification of the error decomposition pipeline.
"""

from typing import Dict, List, Any, Union
from pathlib import Path
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw


# Standard COCO category subset for synthetic verification
SAMPLE_CATEGORIES = [
    {"id": 1, "name": "person", "supercategory": "person"},
    {"id": 3, "name": "car", "supercategory": "vehicle"},
    {"id": 16, "name": "dog", "supercategory": "animal"},
]


def create_synthetic_benchmark(
    output_dir: Union[str, Path] = "sample_data",
    num_images: int = 5,
    seed: int = 42,
) -> Dict[str, Path]:
    """
    Creates a self-contained, offline synthetic dataset with images and annotations
    in official COCO format.

    Args:
        output_dir: Base directory to store images and annotations.
        num_images: Number of sample images to create.
        seed: Random seed for deterministic generation.

    Returns:
        Dict with paths: {'images_dir': Path, 'annotations_path': Path}
    """
    base_dir = Path(output_dir)
    images_dir = base_dir / "images"
    annos_dir = base_dir / "annotations"

    images_dir.mkdir(parents=True, exist_ok=True)
    annos_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)

    images_info: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    anno_id = 1

    width, height = 320, 240

    for i in range(1, num_images + 1):
        image_id = i
        filename = f"{image_id:012d}.jpg"
        img_path = images_dir / filename

        # Create structured image with smooth background
        base_color = rng.randint(90, 180, size=3)
        img_np = np.ones((height, width, 3), dtype=np.uint8) * base_color.astype(np.uint8)
        pil_img = Image.fromarray(img_np, mode="RGB")
        draw = ImageDraw.Draw(pil_img)

        # Place 1 to 3 distinct objects with bounding boxes
        num_objects = rng.randint(1, 4)
        for obj_idx in range(num_objects):
            cat = rng.choice(SAMPLE_CATEGORIES)
            cat_id = int(cat["id"])

            bw = rng.randint(40, 80)
            bh = rng.randint(40, 80)
            bx = rng.randint(10, width - bw - 10)
            by = rng.randint(10, height - bh - 10)

            # Draw colored rectangle to represent the object
            obj_color = (
                rng.randint(30, 240),
                rng.randint(30, 240),
                rng.randint(30, 240),
            )
            draw.rectangle([bx, by, bx + bw, by + bh], fill=obj_color, outline=(255, 255, 255), width=2)

            annotations.append(
                {
                    "id": anno_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [float(bx), float(by), float(bw), float(bh)],
                    "area": float(bw * bh),
                    "iscrowd": 0,
                }
            )
            anno_id += 1

        pil_img.save(img_path, quality=95)

        images_info.append(
            {
                "id": image_id,
                "file_name": filename,
                "width": width,
                "height": height,
            }
        )

    coco_dataset = {
        "images": images_info,
        "annotations": annotations,
        "categories": SAMPLE_CATEGORIES,
    }

    annos_file = annos_dir / "instances_sample.json"
    with open(annos_file, "w") as f:
        json.dump(coco_dataset, f, indent=2)

    return {
        "images_dir": images_dir,
        "annotations_path": annos_file,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare Sample Evaluation Data")
    parser.add_argument("--output-dir", default="sample_data", help="Output directory")
    parser.add_argument("--num-images", type=int, default=5, help="Number of synthetic images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    paths = create_synthetic_benchmark(
        output_dir=args.output_dir,
        num_images=args.num_images,
        seed=args.seed,
    )
    print(f"Sample images created at: {paths['images_dir']}")
    print(f"Annotations saved at  : {paths['annotations_path']}")


if __name__ == "__main__":
    main()
