"""
Unit tests for image_to_text.py — OCR pipeline orchestration.

Run with:  pytest tests/test_image_to_text.py -v
(slow tests are marked with @pytest.mark.slow; run them with: pytest -m slow)
"""

import os
import pytest

SAMPLE_IMAGE = "mau-cv-xin-viec-don-gian-image-1.jpg"


@pytest.mark.slow
class TestImageToText:
    """Tests that require downloading and loading ML models (~200 MB).

    These are marked as slow because the first run will download
    EasyOCR and YOLO model weights.
    """

    @pytest.fixture(scope="class")
    def engine(self):
        """Class-scoped fixture: loads models once for all slow tests."""
        from image_to_text import ImageToText
        return ImageToText()

    def test_instantiation(self, engine):
        """ImageToText should be instantiable without error."""
        assert engine is not None
        assert engine.reader is not None
        assert engine.model is not None
        assert engine.probability is not None

    def test_split_image_does_not_crash(self, engine):
        """split_image should return a list of bounding boxes for the sample image."""
        if not os.path.isfile(SAMPLE_IMAGE):
            pytest.skip(f"Sample image not found: {SAMPLE_IMAGE}")

        result = engine.split_image(SAMPLE_IMAGE)
        assert isinstance(result, list)
        assert len(result) > 0
        # Each box should be [x_min, y_min, x_max, y_max]
        for box in result:
            assert len(box) == 4
            assert box[2] > box[0], "x_max should be > x_min"
            assert box[3] > box[1], "y_max should be > y_min"
