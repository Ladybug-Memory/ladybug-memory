import pytest
import tempfile
import os
from memory import LadybugMemory


@pytest.fixture
def memory():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "memory.lbdb")
        mem = LadybugMemory(db_path)
        yield mem


def test_store_and_search(memory):
    memory.store(
        "User prefers Python for data analysis", memory_type="preference", importance=8
    )
    memory.store("User is allergic to nuts", memory_type="preference", importance=9)

    results = memory.search("python")
    assert len(results) > 0
    assert "python" in results[0].entry.content.lower()


def test_semantic_search(memory):
    memory.store(
        "The python is a large snake found in tropical regions",
        memory_type="fact",
        importance=5,
    )

    results = memory.semantic_search("snake", limit=3)
    assert len(results) > 0
    assert results[0].score > 0


def test_recall_all(memory):
    memory.store(
        "User prefers Python for data analysis", memory_type="preference", importance=8
    )
    memory.store("User is allergic to nuts", memory_type="preference", importance=9)
    memory.store("Fixed bug in login handler", memory_type="work", importance=7)

    entries = memory.recall()
    assert len(entries) == 3


def test_count(memory):
    memory.store(
        "User prefers Python for data analysis", memory_type="preference", importance=8
    )
    memory.store("User is allergic to nuts", memory_type="preference", importance=9)

    assert memory.count() == 2


def test_memory_types(memory):
    memory.store(
        "User prefers Python for data analysis", memory_type="preference", importance=8
    )
    memory.store("Fixed bug in login handler", memory_type="work", importance=7)
    memory.store("The python is a large snake", memory_type="fact", importance=5)

    entries = memory.recall()
    types = {e.memory_type for e in entries}
    assert "preference" in types
    assert "work" in types
    assert "fact" in types
