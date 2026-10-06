import pytest

from src import db, llm


@pytest.fixture(autouse=True)
def fresh_db(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db(reset=True)
    llm.reset_stats()
    yield

