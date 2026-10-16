import tempfile
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from degradations import (
    apply_low_light,
    apply_degradation,
    degrade_directory,
    AVAILABLE_DEGRADATIONS,
    LOW_LIGHT_PARAMS,
)


@pytest.fixture
def sample_rgb_array():
    """Generates a synthetic RGB image (128x128x3) uint8 with varying brightness."""
    rng = np.random.RandomState(42)
    # Mid-to-high brightness image to observe degradation clearly
    arr = rng.randint(80, 240, size=(128, 128, 3), dtype=np.uint8)
    return arr


@pytest.fixture
def sample_pil_image(sample_rgb_array):
    return Image.fromarray(sample_rgb_array, mode="RGB")


def test_registry_contains_low_light():
    assert "low_light" in AVAILABLE_DEGRADATIONS


def test_shape_dtype_and_range_preservation_numpy(sample_rgb_array):
    for sev in range(1, 6):
        out = apply_low_light(sample_rgb_array, severity=sev)
        assert isinstance(out, np.ndarray)
        assert out.shape == sample_rgb_array.shape
        assert out.dtype == np.uint8
        assert np.all(out >= 0) and np.all(out <= 255)


def test_shape_dtype_and_mode_preservation_pil(sample_pil_image):
    for sev in range(1, 6):
        out = apply_low_light(sample_pil_image, severity=sev)
        assert isinstance(out, Image.Image)
        assert out.size == sample_pil_image.size
        assert out.mode == "RGB"


def test_monotonic_luminance_decrease(sample_rgb_array):
    """Verifies that mean image brightness strictly decreases as severity increases from 1 to 5."""
    means = []
    clean_mean = float(np.mean(sample_rgb_array))

    for sev in range(1, 6):
        degraded = apply_low_light(sample_rgb_array, severity=sev)
        deg_mean = float(np.mean(degraded))
        means.append(deg_mean)

    # Clean image must be brighter than all degraded versions
    assert clean_mean > means[0]

    # Monotonic decrease: means[0] > means[1] > ... > means[4]
    for i in range(len(means) - 1):
        assert (
            means[i] > means[i + 1]
        ), f"Severity {i+1} mean ({means[i]:.2f}) is not greater than Severity {i+2} mean ({means[i+1]:.2f})"


def test_original_image_not_modified_in_place(sample_rgb_array):
    copy_orig = sample_rgb_array.copy()
    _ = apply_low_light(sample_rgb_array, severity=3)
    np.testing.assert_array_equal(sample_rgb_array, copy_orig)


def test_invalid_severities_raise_value_error(sample_rgb_array):
    for invalid_sev in [0, 6, -1, 10, "3", 2.5]:
        with pytest.raises(ValueError, match="Severity must be an integer between 1 and 5"):
            apply_low_light(sample_rgb_array, severity=invalid_sev)


def test_unknown_degradation_type_raises_value_error(sample_rgb_array):
    with pytest.raises(ValueError, match="Unknown degradation type"):
        apply_degradation(sample_rgb_array, degradation_type="non_existent", severity=1)


def test_invalid_image_inputs_raise_errors():
    # Wrong dimensions
    with pytest.raises(ValueError, match="Expected 3-channel"):
        apply_low_light(np.zeros((100, 100), dtype=np.uint8), severity=1)

    # Wrong dtype
    with pytest.raises(ValueError, match="must have dtype uint8"):
        apply_low_light(np.zeros((100, 100, 3), dtype=np.float32), severity=1)

    # Wrong type
    with pytest.raises(TypeError, match="Expected image to be PIL.Image.Image or np.ndarray"):
        apply_low_light("invalid_path.jpg", severity=1)


def test_degrade_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        in_dir = tmp_path / "input"
        out_dir = tmp_path / "output"
        in_dir.mkdir()

        # Create two sample images
        img1 = Image.fromarray(np.full((50, 50, 3), 200, dtype=np.uint8), mode="RGB")
        img2 = Image.fromarray(np.full((60, 60, 3), 180, dtype=np.uint8), mode="RGB")
        img1.save(in_dir / "000000000001.jpg")
        img2.save(in_dir / "000000000002.jpg")

        count = degrade_directory(
            input_dir=in_dir,
            output_dir=out_dir,
            degradation_type="low_light",
            severity=3,
        )

        assert count == 2
        assert (out_dir / "000000000001.jpg").is_file()
        assert (out_dir / "000000000002.jpg").is_file()

        with Image.open(out_dir / "000000000001.jpg") as saved_img:
            assert saved_img.size == (50, 50)
            # Must be darker than original 200
            saved_arr = np.array(saved_img)
            assert np.mean(saved_arr) < 200
