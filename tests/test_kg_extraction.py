import pytest
import tempfile
import os
from memory import LadybugMemory


@pytest.fixture
def memory():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "memory.lbdb")
        mem = LadybugMemory(
            db_path=db_path,
            enable_entity_extraction=True,
            gliner_model="fastino/gliner2-base-v1",
            entity_confidence_threshold=0.85,
        )
        yield mem


def test_store_with_entities(memory):
    content = """
        Apple Inc. announced today that Tim Cook will be presenting at their
        headquarters in Cupertino, California. The event is scheduled for March 15, 2024.
        Microsoft and Google are also expected to make announcements during the same week.
        """

    entry, entities, relations_count = memory.store_with_entities(
        content=content,
        memory_type="news",
        importance=8,
        extract_entities=True,
    )

    assert entry is not None
    assert len(entities) > 0


def test_extract_entities(memory):
    text = "Elon Musk founded SpaceX in 2002 and Tesla's headquarters are in Austin, Texas."
    entities = memory.extract_entities(text)

    assert len(entities) > 0
    entity_texts = [e.text for e in entities]
    assert "Elon Musk" in entity_texts or "SpaceX" in entity_texts


def test_search_by_entity(memory):
    content = """
        Apple Inc. announced today that Tim Cook will be presenting at their
        headquarters in Cupertino, California.
        """

    memory.store_with_entities(
        content=content,
        memory_type="news",
        importance=8,
        extract_entities=True,
    )

    results = memory.search_by_entity("Apple Inc.", limit=5)
    assert len(results) > 0


def test_get_all_entities(memory):
    content = """
        Apple Inc. announced today that Tim Cook will be presenting at their
        headquarters in Cupertino, California.
        """

    memory.store_with_entities(
        content=content,
        memory_type="news",
        importance=8,
        extract_entities=True,
    )

    all_entities = memory.get_all_entities(limit=10)
    assert len(all_entities) > 0


def test_entity_graph(memory):
    content = """
        Apple Inc. announced today that Tim Cook will be presenting at their
        headquarters in Cupertino, California.
        """

    memory.store_with_entities(
        content=content,
        memory_type="news",
        importance=8,
        extract_entities=True,
    )

    all_entities = memory.get_all_entities(limit=10)
    if all_entities:
        entity_id = all_entities[0]["id"]
        graph = memory.get_entity_graph(entity_id)
        assert "entity" in graph
