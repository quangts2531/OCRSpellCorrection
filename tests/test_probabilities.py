"""
Unit tests for probabilities.py — N-gram spelling correction module.

Run with:  pytest tests/test_probabilities.py -v
"""

import pytest
from probabilities import Probability


@pytest.fixture(scope="module")
def prob():
    """Module-scoped fixture: loads dictionaries once for all tests."""
    return Probability()


# ---------------------------------------------------------------
# fix_spelling tests
# ---------------------------------------------------------------

class TestFixSpelling:
    def test_empty_string_returns_string(self, prob):
        """fix_spelling should return a string (not None, not crash) for empty input."""
        result = prob.fix_spelling("")
        assert isinstance(result, str)

    def test_single_word(self, prob):
        """fix_spelling should handle a single word without crashing."""
        result = prob.fix_spelling("xin")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_known_ocr_errors(self, prob):
        """fix_spelling should produce non-empty output for a sentence with known OCR errors.

        This uses the same test case from the __main__ block in probabilities.py.
        """
        text = (
            "Có kinh nghiệm lý đội nhóm gổm 20 nhan vien cấp dưởi. "
            "Tính tổ chửc và kỷ luật cao, không ngửng đổi mởi để cải thiện "
            "sản phẩm và dịch vụ đi kèm Luôn đặt trải nghiệm của khách hàng "
            "lên trên cùng và nỗ lực đưa tên tuổi thuong hiệu tiếp can đưạc "
            "tệp người rộng hon. rong quản dung "
        )
        result = prob.fix_spelling(text)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------
# is_valid_token tests
# ---------------------------------------------------------------

class TestIsValidToken:
    def test_pure_integer_returns_false(self, prob):
        assert prob.is_valid_token("12345") is False

    def test_decimal_number_returns_false(self, prob):
        assert prob.is_valid_token("3.14") is False

    def test_email_returns_false(self, prob):
        assert prob.is_valid_token("user@example.com") is False

    def test_phone_number_returns_false(self, prob):
        assert prob.is_valid_token("0901234567") is False

    def test_normal_vietnamese_word_returns_true(self, prob):
        assert prob.is_valid_token("xin") is True

    def test_another_vietnamese_word_returns_true(self, prob):
        assert prob.is_valid_token("chào") is True

    def test_compound_word_with_underscore(self, prob):
        """Compound words like 'kinh_nghiệm' should pass unless detected as proper noun."""
        # We just verify it doesn't crash; NER result may vary
        result = prob.is_valid_token("kinh_nghiệm")
        assert isinstance(result, bool)
