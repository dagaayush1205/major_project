import tempfile
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from degradations import (
    apply_low_light,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_fog,
    apply_jpeg_compression,
    apply_degradation,
    degrade_directory,
    AVAILABLE_DEGRADATIONS,
)


@pytest.fixture
def sample_rgb_array():
    """Generates a synthetic RGB image (128x128x3) uint8 with structured patterns."""
    rng = np.random.RandomState(42)
    # Background
    arr = np.full((128, 128, 3), 120, dtype=np.uint8)
    # Add sharp high-contrast grid patterns for blur and compression detection
    arr[20:108:8, :, :] = 220
    arr[:, 20:108:8, :] = 40
    return arr


@pytest.fixture
def sample_pil_image(sample_rgb_array):
    return Image.fromarray(sample_rgb_array, mode="RGB")


def test_registry_contains_all_degradations():
    expected = {"low_light", "gaussian_blur", "gaussian_noise", "fog", "jpeg_compression"}
    assert expected.issubset(set(AVAILABLE_DEGRADATIONS))


@pytest.mark.parametrize("deg_type", AVAILABLE_DEGRADATIONS)
def test_shape_dtype_and_range_preservation_numpy(sample_rgb_array, deg_type):
    for sev in range(1, 6):
        out = apply_degradation(sample_rgb_array, degradation_type=deg_type, severity=sev, seed=42)
        assert isinstance(out, np.ndarray)
        assert out.shape == sample_rgb_array.shape
        assert out.dtype == np.uint8
        assert np.all(out >= 0) and np.all(out <= 255)


@pytest.mark.parametrize("deg_type", AVAILABLE_DEGRADATIONS)
def test_shape_dtype_and_mode_preservation_pil(sample_pil_image, deg_type):
    for sev in range(1, 6):
        out = apply_degradation(sample_pil_image, degradation_type=deg_type, severity=sev, seed=42)
        assert isinstance(out, Image.Image)
        assert out.size == sample_pil_image.size
        assert out.mode == "RGB"


def test_monotonic_low_light_luminance_decrease(sample_rgb_array):
    """Verifies that mean image brightness strictly decreases as severity increases from 1 to 5."""
    means = []
    clean_mean = float(np.mean(sample_rgb_array))

    for sev in range(1, 6):
        degraded = apply_low_light(sample_rgb_array, severity=sev)
        means.append(float(np.mean(degraded)))

    assert clean_mean > means[0]
    for i in range(len(means) - 1):
        assert means[i] > means[i + 1]


def test_monotonic_gaussian_blur_gradient_decrease(sample_rgb_array):
    """Verifies that high-frequency gradient energy strictly decreases as blur severity increases."""
    def high_freq_energy(img_np):
        # Difference between adjacent pixels
        diff_y = np.diff(img_np.astype(np.float32), axis=0)
        diff_x = np.diff(img_np.astype(np.float32), axis=1)
        return float(np.mean(diff_y ** 2) + np.mean(diff_x ** 2))

    clean_hf = high_freq_energy(sample_rgb_array)
    hf_energies = []

    for sev in range(1, 6):
        blurred = apply_gaussian_blur(sample_rgb_array, severity=sev)
        hf_energies.append(high_freq_energy(blurred))

    assert clean_hf > hf_energies[0]
    for i in range(len(hf_energies) - 1):
        assert hf_energies[i] > hf_energies[i + 1]


def test_monotonic_gaussian_noise_mse_increase(sample_rgb_array):
    """Verifies that Mean Squared Error (MSE) strictly increases as noise severity increases."""
    mses = []
    for sev in range(1, 6):
        noisy = apply_gaussian_noise(sample_rgb_array, severity=sev, seed=123)
        mse = float(np.mean((noisy.astype(np.float32) - sample_rgb_array.astype(np.float32)) ** 2))
        mses.append(mse)

    for i in range(len(mses) - 1):
        assert mses[i] < mses[i + 1]


def test_gaussian_noise_seed_repeatability(sample_rgb_array):
    out1 = apply_gaussian_noise(sample_rgb_array, severity=3, seed=999)
    out2 = apply_gaussian_noise(sample_rgb_array, severity=3, seed=999)
    out3 = apply_gaussian_noise(sample_rgb_array, severity=3, seed=888)

    np.testing.assert_array_equal(out1, out2)
    assert not np.array_equal(out1, out3)


def test_monotonic_fog_contrast_decrease(sample_rgb_array):
    """Verifies that image contrast (std dev) strictly decreases as fog severity increases."""
    stds = []
    clean_std = float(np.std(sample_rgb_array))

    for sev in range(1, 6):
        foggy = apply_fog(sample_rgb_array, severity=sev)
        stds.append(float(np.std(foggy)))

    assert clean_std > stds[0]
    for i in range(len(stds) - 1):
        assert stds[i] > stds[i + 1]


def test_monotonic_jpeg_compression_mse_increase(sample_rgb_array):
    """Verifies that compression error (MSE) strictly increases as JPEG quality decreases (sev 1 to 5)."""
    mses = []
    for sev in range(1, 6):
        comp = apply_jpeg_compression(sample_rgb_array, severity=sev)
        mse = float(np.mean((comp.astype(np.float32) - sample_rgb_array.astype(np.float32)) ** 2))
        mses.append(mse)

    for i in range(len(mses) - 1):
        assert mses[i] < mses[i + 1]


def test_original_image_not_modified_in_place(sample_rgb_array):
    for deg in AVAILABLE_DEGRADATIONS:
        copy_orig = sample_rgb_array.copy()
        _ = apply_degradation(sample_rgb_array, degradation_type=deg, severity=3)
        np.testing.assert_array_equal(sample_rgb_array, copy_orig)


def test_invalid_severities_raise_value_error(sample_rgb_array):
    for invalid_sev in [0, 6, -1, 10, "3", 2.5]:
        with pytest.raises(ValueError, match="Severity must be an integer between 1 and 5"):
            apply_low_light(sample_rgb_array, severity=invalid_sev)


def test_unknown_degradation_type_raises_value_error(sample_rgb_array):
    with pytest.raises(ValueError, match="Unknown degradation type"):
        apply_degradation(sample_rgb_array, degradation_type="non_existent", severity=1)


def test_invalid_image_inputs_raise_errors():
    with pytest.raises(ValueError, match="Expected 3-channel"):
        apply_low_light(np.zeros((100, 100), dtype=np.uint8), severity=1)

    with pytest.raises(ValueError, match="must have dtype uint8"):
        apply_low_light(np.zeros((100, 100, 3), dtype=np.float32), severity=1)

    with pytest.raises(TypeError, match="Expected image to be PIL.Image.Image or np.ndarray"):
        apply_low_light("invalid_path.jpg", severity=1)


def test_degrade_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        in_dir = tmp_path / "input"
        out_dir = tmp_path / "output"
        in_dir.mkdir()

        img1 = Image.fromarray(np.full((50, 50, 3), 200, dtype=np.uint8), mode="RGB")
        img2 = Image.fromarray(np.full((60, 60, 3), 180, dtype=np.uint8), mode="RGB")
        img1.save(in_dir / "000000000001.jpg")
        img2.save(in_dir / "000000000002.jpg")

        count = degrade_directory(
            input_dir=in_dir,
            output_dir=out_dir,
            degradation_type="gaussian_blur",
            severity=3,
        )

        assert count == 2
        assert (out_dir / "000000000001.jpg").is_file()
        assert (out_dir / "000000000002.jpg").is_file()
