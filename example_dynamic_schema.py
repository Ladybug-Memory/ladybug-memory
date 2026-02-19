#!/usr/bin/env python3
"""Example demonstrating dynamic schema discovery using icebug parallel leiden clustering."""

from memory import LadybugMemory, DynamicSchemaDiscovery


def main():
    # Initialize memory with entity extraction enabled
    memory = LadybugMemory(
        db_path="memory.lbdb",
        enable_entity_extraction=True,
        gliner_model="fastino/gliner2-base-v1",
        entity_confidence_threshold=0.85,
    )

    print("=" * 60)
    print("Dynamic Schema Discovery Example")
    print("=" * 60)
    print("\nThis example demonstrates:")
    print("1. Extracting entities without predefined schema")
    print("2. Building entity similarity graph from embeddings")
    print("3. Running icebug parallel leiden clustering")
    print("4. Discovering schema types from clusters")
    print("5. Creating dynamic tables for high-confidence schemas")

    # Example 1: Store multiple memories with entity extraction
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

    print("\n1. Extracting entities from multiple memories...")
    for i, content in enumerate(contents, 1):
        print(f"   Processing memory {i}/{len(contents)}...")
        entry, entities = memory.store_with_entities(
            content=content,
            memory_type="news",
            importance=8,
            extract_entities=True,
        )
        all_entities.extend(entities)

    print(f"\n   Total entities extracted: {len(all_entities)}")
    print("   Sample entities:")
    for entity in all_entities[:10]:
        print(f"   - {entity.text} (confidence: {entity.confidence:.3f})")

    # Example 2: Run dynamic schema discovery
    print("\n2. Running dynamic schema discovery with icebug parallel leiden...")

    schema_discovery = DynamicSchemaDiscovery(
        similarity_threshold=0.75,
        min_cluster_size=2,
        min_confidence=0.6,
    )

    discovered_schemas, entity_to_type = schema_discovery.discover_schema(all_entities)

    print(f"\n   Discovered {len(discovered_schemas)} schema types:")
    for schema in discovered_schemas:
        print(f"\n   Schema: {schema.type_name}")
        print(f"   - Confidence: {schema.confidence:.3f}")
        print(f"   - Cluster size: {schema.size} entities")
        print(f"   - Sample entities: {', '.join(schema.sample_entities[:3])}")

    # Example 3: Create dynamic tables
    print("\n3. Creating dynamic tables for discovered schemas...")

    # Convert to dict format for the method
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

    table_mapping = memory.create_dynamic_schema_tables(schema_dicts)

    print(f"\n   Created {len(table_mapping)} dynamic tables:")
    for type_name, table_name in table_mapping.items():
        print(f"   - {type_name} -> {table_name}")

    # Example 4: Store with dynamic schema in one call
    print("\n4. Storing memory with automatic dynamic schema discovery...")

    new_content = """
    OpenAI CEO Sam Altman announced GPT-5 at their San Francisco headquarters.
    The announcement was made on February 20, 2024, alongside other AI researchers.
    """

    entry, results = memory.store_with_dynamic_schema(
        content=new_content,
        memory_type="tech_news",
        importance=9,
    )

    print(f"\n   Stored memory ID: {entry.id}")
    print(f"   Entities extracted: {len(results['entities'])}")
    print(f"   Discovered schemas: {len(results['discovered_schemas'])}")
    print(f"   Table mapping: {results['table_mapping']}")

    print("\n5. Summary of discovered schemas:")
    for schema in results["discovered_schemas"]:
        print(f"   - {schema['type_name']} (confidence: {schema['confidence']:.3f})")

    print("\n" + "=" * 60)
    print("Dynamic schema discovery completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
