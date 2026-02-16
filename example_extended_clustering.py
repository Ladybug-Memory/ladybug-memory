#!/usr/bin/env python3
"""Extended example demonstrating dynamic schema discovery with more entities."""

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
    print("Extended Dynamic Schema Discovery Example")
    print("=" * 60)

    # Extended example: Store more memories with diverse entities
    # Using 4 memories to keep clustering fast while testing diverse entity types
    contents = [
        # Tech companies and CEOs
        """Apple Inc. announced today that Tim Cook will be presenting at their 
        headquarters in Cupertino, California. The event is scheduled for March 15, 2024.
        Microsoft CEO Satya Nadella will also attend the conference.""",
        # More tech companies
        """Google LLC revealed new AI products at their Mountain View campus.
        Sundar Pichai, CEO of Google, demonstrated Gemini to the press.
        Amazon's Andy Jassy was also present at the tech showcase.
        The event took place on January 10, 2024.""",
        # Automotive and space
        """Tesla Inc. unveiled the new Model 3 in Austin, Texas.
        Elon Musk presented alongside other executives from SpaceX.
        The event took place on January 10, 2024.""",
        # Finance and banking
        """JPMorgan Chase CEO Jamie Dimon spoke at the World Economic Forum in Davos, Switzerland.
        The discussion covered global economic trends and banking regulations.
        Christine Lagarde of the European Central Bank also participated.
        The event occurred on February 20, 2024.""",
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

    # Group by type
    type_counts = {}
    for entity in all_entities:
        etype = entity.entity_type
        type_counts[etype] = type_counts.get(etype, 0) + 1

    print("   Entity types from GLiNER2:")
    for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"   - {etype}: {count} entities")

    # Example 2: Run dynamic schema discovery
    print("\n2. Running dynamic schema discovery with icebug parallel leiden...")

    # Use relaxed parameters for faster convergence with larger entity sets
    schema_discovery = DynamicSchemaDiscovery(
        similarity_threshold=0.65,  # Lower threshold for more edges but faster
        min_cluster_size=2,
        min_confidence=0.5,  # Lower confidence for more schemas
    )

    discovered_schemas, entity_to_type = schema_discovery.discover_schema(all_entities)

    print(f"\n   Discovered {len(discovered_schemas)} schema types:")
    for schema in discovered_schemas:
        print(f"\n   Schema: {schema.type_name}")
        print(f"   - Confidence: {schema.confidence:.3f}")
        print(f"   - Cluster size: {schema.size} entities")
        print(f"   - Sample entities: {', '.join(schema.sample_entities[:5])}")

    # Example 3: Create dynamic tables
    print("\n3. Creating dynamic tables for discovered schemas...")

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

    # Summary statistics
    print("\n4. Clustering Analysis:")
    print(f"   - Total entities: {len(all_entities)}")
    print(f"   - Discovered schemas: {len(discovered_schemas)}")

    # Check if locations are properly clustered
    location_entities = [e for e in all_entities if e.entity_type == "location"]
    print(f"   - Total locations: {len(location_entities)}")
    if location_entities:
        location_clusters = set()
        for schema in discovered_schemas:
            if schema.type_name == "location":
                location_clusters.add(schema.cluster_id)
        print(f"   - Location clusters: {len(location_clusters)}")

    print("\n" + "=" * 60)
    print("Extended dynamic schema discovery completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
