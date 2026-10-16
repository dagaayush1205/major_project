from pathlib import Path
import pytest

from detect import get_image_id, COCO80_TO_COCO91, load_config


def test_coco80_to_coco91_mapping_validity():
    # Exactly 80 categories corresponding to YOLO class indices 0..79
    assert len(COCO80_TO_COCO91) == 80

    # All values must be valid official COCO 91 category IDs (range 1 to 90)
    for cat_id in COCO80_TO_COCO91:
        assert isinstance(cat_id, int)
        assert 1 <= cat_id <= 90

    # Ensure no duplicates
    assert len(set(COCO80_TO_COCO91)) == 80


def test_get_image_id_numeric_stem():
    assert get_image_id("000000397133.jpg") == 397133
    assert get_image_id("/path/to/000000000042.png") == 42
    assert get_image_id(Path("val2017/12345.jpg")) == 12345


def test_get_image_id_non_numeric_stem_raises_value_error():
    with pytest.raises(ValueError, match="does not contain a numeric COCO image ID"):
        get_image_id("cat_photo.jpg")

    with pytest.raises(ValueError, match="does not contain a numeric COCO image ID"):
        get_image_id("/path/to/frame_001_a.png")


def test_load_config_defaults():
    config = load_config("config.yaml")
    assert "inference" in config
    assert "models" in config
    assert "confidence_threshold" in config["inference"]
    assert "max_detections" in config["inference"]
    assert config["inference"]["confidence_threshold"] == 0.01
    assert config["inference"]["max_detections"] == 300

    # Models present
    assert "yolo11" in config["models"]
    assert "fasterrcnn" in config["models"]
    assert "rtdetr" in config["models"]
