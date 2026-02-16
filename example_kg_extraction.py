#!/usr/bin/env python3
"""Example demonstrating GLiNER2 entity extraction with ladybug-memory."""

from memory import LadybugMemory, GLiNEREntityExtractor


def main():
    # Initialize memory with entity extraction enabled
    memory = LadybugMemory(
        db_path="memory.lbdb",
        enable_entity_extraction=True,
        gliner_model="fastino/gliner2-base-v1",  # GLiNER2 model
        entity_confidence_threshold=0.85,
    )

    print("=" * 60)
    print("GLiNER2 Entity Extraction Example")
    print("=" * 60)

    # Example 1: Store content with entity extraction
    content = """
        Apple Inc. announced today that Tim Cook will be presenting at their 
        headquarters in Cupertino, California. The event is scheduled for March 15, 2024.
        Microsoft and Google are also expected to make announcements during the same week.
        """

    print("\n1. Storing memory with entity extraction:")
    print(f"   Content: {content[:100]}...")

    entry, entities = memory.store_with_entities(
        content=content,
        memory_type="news",
        importance=8,
        extract_entities=True,
    )

    print(f"\n   Stored memory ID: {entry.id}")
    print(f"   Extracted {len(entities)} entities:")
    for entity in entities:
        print(
            f"   - {entity.text} ({entity.entity_type}, confidence: {entity.confidence:.3f})"
        )

    # Example 2: Direct entity extraction
    print("\n2. Direct entity extraction from text:")
    text = "Elon Musk founded SpaceX in 2002 and Tesla's headquarters are in Austin, Texas."
    entities = memory.extract_entities(text)

    print(f"   Text: {text}")
    print(f"   Extracted entities:")
    for entity in entities:
        print(
            f"   - {entity.text} ({entity.entity_type}, confidence: {entity.confidence:.3f})"
        )

    # Example 3: Search by entity
    print("\n3. Searching memories by entity 'Apple Inc.':")
    results = memory.search_by_entity("Apple Inc.", limit=5)
    print(f"   Found {len(results)} memories mentioning 'Apple Inc.'")
    for result in results:
        print(
            f"   - Score: {result.score:.3f}, Content: {result.entry.content[:80]}..."
        )

    # Example 4: Get entity graph
    if entities:
        print("\n4. Entity graph exploration:")
        # First, get all entities
        all_entities = memory.get_all_entities(limit=10)
        if all_entities:
            entity_id = all_entities[0]["id"]
            entity_name = all_entities[0]["canonical_name"]
            print(f"   Exploring entity: {entity_name}")

            graph = memory.get_entity_graph(entity_id)
            print(f"   Entity data: {graph.get('entity', {})}")
            print(f"   Related entities: {len(graph.get('related_entities', []))}")
            print(f"   Mentioned in {len(graph.get('mentioned_in', []))} memories")

    # Example 5: List all entities
    print("\n5. All entities in knowledge graph:")
    all_entities = memory.get_all_entities(limit=20)
    print(f"   Total entities: {len(all_entities)}")
    for entity in all_entities[:5]:  # Show first 5
        print(f"   - {entity['canonical_name']} ({entity['entity_type']})")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
