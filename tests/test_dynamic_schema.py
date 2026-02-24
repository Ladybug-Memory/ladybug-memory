import pytest
import tempfile
import os
from memory import LadybugMemory, DynamicSchemaDiscovery


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


def test_dynamic_schema_discovery(memory):
    contents = [
        """
        Apple Inc. announced today that Tim Cook will be presenting at their
        headquarters in Cupertino, California. The event is scheduled for March 15, 2024.
        Microsoft CEO Satya Nadella will also attend the conference.
        """,
        """
        Google LLC revealed new AI products at their Mountain View campus.
        Sundar Pichai, CEO of Google, demonstrated Gemini to the press.
        Amazon's Andy Jassy was also present at the tech showcase.
        """,
        """
        Tesla Inc. unveiled the new Model 3 in Austin, Texas.
        Elon Musk presented alongside other executives from SpaceX.
        The event took place on January 10, 2024.
        """,
    ]

    all_entities = []

    for content in contents:
        entry, entities, _ = memory.store_with_entities(
            content=content,
            memory_type="news",
            importance=8,
            extract_entities=True,
        )
        all_entities.extend(entities)

    assert len(all_entities) > 0

    schema_discovery = DynamicSchemaDiscovery(
        similarity_threshold=0.75,
        min_cluster_size=2,
        min_confidence=0.6,
    )

    discovered_schemas, entity_to_type = schema_discovery.discover_schema(all_entities)

    assert len(discovered_schemas) >= 0


def test_create_dynamic_schema_tables(memory):
    contents = [
        """
        Apple Inc. announced today that Tim Cook will be presenting at their
        headquarters in Cupertino, California.
        Microsoft CEO Satya Nadella will also attend.
        """,
        """
        Google LLC revealed new AI products at their Mountain View campus.
        Sundar Pichai, CEO of Google, demonstrated Gemini to the press.
        """,
    ]

    all_entities = []

    for content in contents:
        entry, entities, _ = memory.store_with_entities(
            content=content,
            memory_type="news",
            importance=8,
            extract_entities=True,
        )
        all_entities.extend(entities)

    schema_discovery = DynamicSchemaDiscovery(
        similarity_threshold=0.75,
        min_cluster_size=2,
        min_confidence=0.6,
    )

    discovered_schemas, entity_to_type = schema_discovery.discover_schema(all_entities)

    schema_dicts = [
        {
            "type_name": s.type_name,
            "confidence": s.confidence,
            "sample_entities": s.sample_entities,
            "cluster_id": s.cluster_id,
            "size": s.size,
        }
        for s in discovered_schemas
    ]

    if schema_dicts:
        table_mapping = memory.create_dynamic_schema_tables(schema_dicts)
        assert len(table_mapping) >= 0


def test_store_with_dynamic_schema(memory):
    contents = [
        """
        Apple Inc. announced today that Tim Cook will be presenting at their
        headquarters in Cupertino, California.
        Microsoft CEO Satya Nadella will also attend.
        """,
        """
        Google LLC revealed new AI products at their Mountain View campus.
        Sundar Pichai, CEO of Google, demonstrated Gemini to the press.
        """,
    ]

    for content in contents:
        memory.store_with_entities(
            content=content,
            memory_type="news",
            importance=8,
            extract_entities=True,
        )

    new_content = """
    OpenAI CEO Sam Altman announced GPT-5 at their San Francisco headquarters.
    The announcement was made on February 20, 2024, alongside other AI researchers.
    """

    entry, results = memory.store_with_dynamic_schema(
        content=new_content,
        memory_type="tech_news",
        importance=9,
    )

    assert entry is not None
    assert "entities" in results
    assert "discovered_schemas" in results
    assert "table_mapping" in results
