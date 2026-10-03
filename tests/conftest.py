"""Общие настройки тестов."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_custom_basis_dir(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Каталог пользовательских базисов не должен зависеть от домашнего каталога разработчика."""
    directory: Path = tmp_path_factory.mktemp("custom-basis")
    monkeypatch.setenv("QUANTUMLAB_BASIS_DIR", str(directory))
