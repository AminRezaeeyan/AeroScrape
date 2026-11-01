import importlib

import pytest

MODULES = [
    "aeroscrape.analysis.association",
    "aeroscrape.cli",
    "aeroscrape.config",
    "aeroscrape.data.clean",
    "aeroscrape.data.validate",
    "aeroscrape.features.preprocess",
    "aeroscrape.models.train",
    "aeroscrape.models.tune",
    "aeroscrape.pipeline.tasks",
    "aeroscrape.scraping.scraper",
    "aeroscrape.serving.api",
    "aeroscrape.storage.database",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module: str) -> None:
    importlib.import_module(module)
