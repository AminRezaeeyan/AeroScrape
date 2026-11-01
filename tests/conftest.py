from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_raw_path() -> Path:
    return FIXTURES / "sample_raw.csv"
