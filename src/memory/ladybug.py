from datetime import datetime
from typing import Any, cast
import uuid

import real_ladybug as lb
from fastembed import TextEmbedding

from memory.chunker import LogicalChunker
from memory.entities import Entity
from memory.extraction import GLiNEREntityExtractor
from memory.interface import (
    AgentMemory,
    MemoryEntry,
    MemorySearchResult,
)


def _get_result(result: lb.QueryResult | list[lb.QueryResult]) -> lb.QueryResult:
    if isinstance(result, list):
        return result[0]
    return result


class LadybugMemory(AgentMemory):
    def __init__(
        self,
        db_path: str,
        enable_entity_extraction: bool = True,
        gliner_model: str = "fastino/gliner2-base-v1",
        entity_confidence_threshold: float = 0.85,
    ):
        self.db = lb.Database(db_path)
        self.conn = lb.Connection(self.db)
        self._init_schema()
        self._init_fts_index()
        self._init_vector_search()

        # Entity extraction
        self._enable_entity_extraction = enable_entity_extraction
        self._entity_extractor: GLiNEREntityExtractor | None = None
        if enable_entity_extraction:
            self._entity_extractor = GLiNEREntityExtractor(
                model_name=gliner_model,
                confidence_threshold=entity_confidence_threshold,
            )

    def _init_schema(self) -> None:
        self.conn.execute("INSTALL JSON; LOAD EXTENSION JSON;")
        self.conn.execute("INSTALL FTS; LOAD EXTENSION FTS;")
        self.conn.execute("INSTALL vector; LOAD EXTENSION vector;")
        self.conn.execute(
            """
            CREATE NODE TABLE IF NOT EXISTS Memory(
                id SERIAL PRIMARY KEY,
                content STRING,
                memory_type STRING,
                importance INT64,
                metadata JSON,
                embedding FLOAT[384],
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE REL TABLE IF NOT EXISTS MemoryLink(
                FROM Memory TO Memory,
                relation STRING
            )
            """
        )
        # Entity schema for knowledge graph
        self._init_entity_schema()

    def _init_entity_schema(self) -> None:
        """Initialize entity and mention tables for knowledge graph."""
        # Entity nodes (canonical entities after disambiguation)
        self.conn.execute(
            """
            CREATE NODE TABLE IF NOT EXISTS Entity(
                id SERIAL PRIMARY KEY,
                canonical_name STRING,
                entity_type STRING,
                embedding FLOAT[384],
                metadata JSON,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
            """
        )
        # Entity mention edges (links entities to memories)
        self.conn.execute(
            """
            CREATE REL TABLE IF NOT EXISTS MentionedIn(
                FROM Entity TO Memory,
                mention_text STRING,
                confidence FLOAT,
                span_start INT64,
                span_end INT64
            )
            """
        )
        # Coreference edges (links similar entities)
        self.conn.execute(
            """
            CREATE REL TABLE IF NOT EXISTS Coreference(
                FROM Entity TO Entity,
                similarity_score FLOAT
            )
            """
        )
        # Relations edges (extracted relations between entities)
        self.conn.execute(
            """
            CREATE REL TABLE IF NOT EXISTS Relations(
                FROM Entity TO Entity,
                relation_type STRING,
                confidence FLOAT,
                metadata JSON
            )
            """
        )
        # Relations edges (extracted relations between entities)
        self.conn.execute(
            """
            CREATE REL TABLE IF NOT EXISTS Relations(
                FROM Entity TO Entity,
                relation_type STRING,
                confidence FLOAT,
                metadata JSON
            )
            """
        )
        # Relations edges (extracted relations between entities)
        self.conn.execute(
            """
            CREATE REL TABLE IF NOT EXISTS Relations(
                FROM Entity TO Entity,
                relation_type STRING,
                confidence FLOAT,
                source_text STRING,
                target_text STRING,
                metadata JSON
            )
            """
        )
        # Discovered schema types (tracks hierarchy: specific_type IS_A base_type)
        self.conn.execute(
            """
            CREATE NODE TABLE IF NOT EXISTS DiscoveredSchemaType(
                id SERIAL PRIMARY KEY,
                type_name STRING,
                base_type STRING,
                confidence FLOAT,
                sample_entities JSON,
                entity_count INT64,
                created_at TIMESTAMP
            )
            """
        )

    def _init_fts_index(self) -> None:
        try:
            self.conn.execute(
                """
                CALL CREATE_FTS_INDEX(
                    'Memory',
                    'memory_content_fts',
                    ['content'],
                    stemmer := 'porter'
                )
                """
            )
        except Exception:
            pass

    def _init_embedding_model(self) -> None:
        self._embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")

    def _get_embedding(self, text: str) -> list[float]:
        if not hasattr(self, "_embedding_model"):
            self._init_embedding_model()
        embeddings = list(self._embedding_model.embed([text]))
        emb = embeddings[0]
        if hasattr(emb, "tolist"):
            return emb.tolist()
        return list(emb)

    def _init_vector_search(self) -> None:
        try:
            self.conn.execute(
                """
                CALL CREATE_VECTOR_INDEX(
                    'Memory',
                    'memory_content_index',
                    'embedding',
                    metric := 'cosine'
                )
                """
            )
        except Exception:
            pass

    def _row_to_entry(self, row: list | dict) -> MemoryEntry:
        if isinstance(row, dict):
            return MemoryEntry(
                id=cast(int, row.get("id")),
                content=cast(str, row.get("content")),
                memory_type=cast(str, row.get("memory_type")),
                importance=cast(int, row.get("importance")),
                metadata=row.get("metadata"),
                created_at=cast(datetime, row.get("created_at")),
                updated_at=cast(datetime, row.get("updated_at")),
            )
        return MemoryEntry(
            id=int(row[0]),
            content=str(row[1]),
            memory_type=str(row[2]),
            importance=int(row[3]),
            metadata=row[4],
            created_at=row[5] if isinstance(row[5], datetime) else None,
            updated_at=row[6] if isinstance(row[6], datetime) else None,
        )

    def store(
        self,
        content: str,
        memory_type: str = "general",
        importance: int = 5,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        now = datetime.now()
        embedding = self._get_embedding(content)

        parameters = {
            "content": content,
            "memory_type": memory_type,
            "importance": importance,
            "embedding": embedding,
            "created_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        }

        if metadata:
            import json

            parameters["metadata"] = json.dumps(metadata)
            raw_result = self.conn.execute(
                """
                CREATE (m:Memory {
                    content: $content,
                    memory_type: $memory_type,
                    importance: $importance,
                    metadata: CAST($metadata AS JSON),
                    embedding: $embedding,
                    created_at: timestamp($created_at),
                    updated_at: timestamp($updated_at)
                })
                RETURN m.id
                """,
                parameters=parameters,
            )
        else:
            raw_result = self.conn.execute(
                """
                CREATE (m:Memory {
                    content: $content,
                    memory_type: $memory_type,
                    importance: $importance,
                    embedding: $embedding,
                    created_at: timestamp($created_at),
                    updated_at: timestamp($updated_at)
                })
                RETURN m.id
                """,
                parameters=parameters,
            )

        result = _get_result(raw_result)
        row = result.get_next()
        memory_id = int(row[0])

        return MemoryEntry(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            metadata=metadata,
            created_at=now,
            updated_at=now,
        )

    def search(
        self,
        query: str,
        limit: int = 5,
        memory_type: str | None = None,
    ) -> list[MemorySearchResult]:
        if memory_type:
            cypher = f"""
                CALL QUERY_FTS_INDEX('Memory', 'memory_content_fts', '{query.replace("'", "''")}', top := {limit})
                WITH node AS m, score
                WHERE m.memory_type = '{memory_type}'
                RETURN m.id, m.content, m.memory_type, m.importance, m.metadata, m.created_at, m.updated_at, score
                ORDER BY score DESC
            """
        else:
            cypher = f"""
                CALL QUERY_FTS_INDEX('Memory', 'memory_content_fts', '{query.replace("'", "''")}', top := {limit})
                WITH node AS m, score
                RETURN m.id, m.content, m.memory_type, m.importance, m.metadata, m.created_at, m.updated_at, score
                ORDER BY score DESC
            """

        raw_result = self.conn.execute(cypher)
        result = _get_result(raw_result)
        search_results = []

        while result.has_next():
            row = result.get_next()
            entry = self._row_to_entry(row)
            score = float(row[7]) if len(row) > 7 else 1.0
            search_results.append(MemorySearchResult(entry=entry, score=score))

        return search_results

    def semantic_search(
        self,
        query: str,
        limit: int = 5,
        memory_type: str | None = None,
    ) -> list[MemorySearchResult]:
        query_embedding = self._get_embedding(query)
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        if memory_type:
            cypher = f"""
                CALL QUERY_VECTOR_INDEX(
                    'Memory',
                    'memory_content_index',
                    {embedding_str},
                    {limit}
                )
                WITH node AS m, distance
                WHERE m.memory_type = '{memory_type}'
                RETURN m.id, m.content, m.memory_type, m.importance, m.metadata, m.created_at, m.updated_at, distance
                ORDER BY distance
            """
        else:
            cypher = f"""
                CALL QUERY_VECTOR_INDEX(
                    'Memory',
                    'memory_content_index',
                    {embedding_str},
                    {limit}
                )
                WITH node AS m, distance
                RETURN m.id, m.content, m.memory_type, m.importance, m.metadata, m.created_at, m.updated_at, distance
                ORDER BY distance
            """

        raw_result = self.conn.execute(cypher)
        result = _get_result(raw_result)
        search_results = []

        while result.has_next():
            row = result.get_next()
            entry = self._row_to_entry(row)
            distance = float(row[7]) if len(row) > 7 else 0.0
            score = 1.0 / (1.0 + distance)
            search_results.append(MemorySearchResult(entry=entry, score=score))

        return search_results

    def recall(
        self,
        limit: int = 10,
        min_importance: int = 0,
        memory_type: str | None = None,
    ) -> list[MemoryEntry]:
        if memory_type:
            cypher = f"""
                MATCH (m:Memory)
                WHERE m.memory_type = '{memory_type}'
                AND m.importance >= {min_importance}
                RETURN m.id, m.content, m.memory_type, m.importance, m.metadata, m.created_at, m.updated_at
                ORDER BY m.importance DESC, m.created_at DESC
                LIMIT {limit}
            """
        else:
            cypher = f"""
                MATCH (m:Memory)
                WHERE m.importance >= {min_importance}
                RETURN m.id, m.content, m.memory_type, m.importance, m.metadata, m.created_at, m.updated_at
                ORDER BY m.importance DESC, m.created_at DESC
                LIMIT {limit}
            """

        raw_result = self.conn.execute(cypher)
        result = _get_result(raw_result)
        entries = []

        while result.has_next():
            row = result.get_next()
            entries.append(self._row_to_entry(row))

        return entries

    def get(self, memory_id: str) -> MemoryEntry | None:
        cypher = f"""
            MATCH (m:Memory)
            WHERE m.id = {memory_id}
            RETURN m.id, m.content, m.memory_type, m.importance, m.metadata, m.created_at, m.updated_at
        """

        raw_result = self.conn.execute(cypher)
        result = _get_result(raw_result)

        if result.has_next():
            row = result.get_next()
            return self._row_to_entry(row)

        return None

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        importance: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry | None:
        updates = []

        if content is not None:
            escaped_content = content.replace("'", "''")
            updates.append(f"m.content = '{escaped_content}'")
        if importance is not None:
            updates.append(f"m.importance = {importance}")
        if metadata is not None:
            updates.append(f"m.metadata = CAST('{str(metadata)}' AS JSON)")

        if not updates:
            return self.get(memory_id)

        now = datetime.now()
        updates.append(
            f"m.updated_at = timestamp('{now.strftime('%Y-%m-%d %H:%M:%S')}')"
        )

        cypher = f"""
            MATCH (m:Memory)
            WHERE m.id = {memory_id}
            SET {", ".join(updates)}
            RETURN m.id, m.content, m.memory_type, m.importance, m.metadata, m.created_at, m.updated_at
        """

        raw_result = self.conn.execute(cypher)
        result = _get_result(raw_result)

        if result.has_next():
            row = result.get_next()
            return self._row_to_entry(row)

        return None

    def delete(self, memory_id: str) -> bool:
        cypher = f"""
            MATCH (m:Memory)
            WHERE m.id = {memory_id}
            DELETE m
        """

        self.conn.execute(cypher)
        return True

    def link(
        self,
        source_id: str,
        target_id: str,
        relation: str = "related",
    ) -> bool:
        cypher = f"""
            MATCH (a:Memory), (b:Memory)
            WHERE a.id = {source_id} AND b.id = {target_id}
            CREATE (a)-[r:MemoryLink {{relation: '{relation}'}}]->(b)
        """

        self.conn.execute(cypher)
        return True

    def get_related(
        self,
        memory_id: str,
        relation: str | None = None,
        max_depth: int = 1,
    ) -> list[tuple[MemoryEntry, str]]:
        if relation:
            cypher = f"""
                MATCH (m:Memory)-[r:MemoryLink {{relation: '{relation}'}}]->(related:Memory)
                WHERE m.id = {memory_id}
                RETURN related.id, related.content, related.memory_type, related.importance, related.metadata, related.created_at, related.updated_at, r.relation
            """
        else:
            cypher = f"""
                MATCH (m:Memory)-[r:MemoryLink]->(related:Memory)
                WHERE m.id = {memory_id}
                RETURN related.id, related.content, related.memory_type, related.importance, related.metadata, related.created_at, related.updated_at, r.relation
            """

        raw_result = self.conn.execute(cypher)
        result = _get_result(raw_result)
        related = []

        while result.has_next():
            row = result.get_next()
            entry = MemoryEntry(
                id=str(row[0]),
                content=str(row[1]),
                memory_type=str(row[2]),
                importance=int(row[3]),
                metadata=row[4],
                created_at=row[5] if isinstance(row[5], datetime) else None,
                updated_at=row[6] if isinstance(row[6], datetime) else None,
            )
            relation_type = str(row[7])
            related.append((entry, relation_type))

        return related

    def count(self) -> int:
        cypher = """
            MATCH (m:Memory)
            RETURN count(m)
        """

        raw_result = self.conn.execute(cypher)
        result = _get_result(raw_result)

        if result.has_next():
            row = result.get_next()
            return int(row[0])

        return 0

    # Entity extraction methods
    def extract_entities(
        self,
        content: str,
        labels: list[str] | None = None,
        threshold: float | None = None,
    ) -> list[Entity]:
        """Extract entities from content using GLiNER2.

        Args:
            content: The text to extract entities from
            labels: Entity types to extract (uses defaults if None)
            threshold: Confidence threshold (uses default if None)

        Returns:
            List of extracted entities
        """
        if not self._entity_extractor:
            raise RuntimeError(
                "Entity extraction is not enabled. Initialize with enable_entity_extraction=True"
            )

        return self._entity_extractor.extract_with_context(
            content, labels=labels, threshold=threshold
        )

    def store_with_entities(
        self,
        content: str,
        memory_type: str = "general",
        importance: int = 5,
        metadata: dict[str, Any] | None = None,
        extract_entities: bool = True,
    ) -> tuple[MemoryEntry, list[Entity], int]:
        """Store memory and optionally extract/link entities and relations.

        Uses H-GLUE logical chunking to improve relation extraction by
        processing semantically coherent units rather than entire documents.

        Args:
            content: The memory content
            memory_type: Type of memory
            importance: Importance score
            metadata: Additional metadata
            extract_entities: Whether to extract and link entities

        Returns:
            Tuple of (MemoryEntry, list of extracted entities, relations_count)
        """
        entry = self.store(content, memory_type, importance, metadata)

        entities: list[Entity] = []
        relations_count = 0
        if extract_entities and self._entity_extractor:
            chunker = LogicalChunker()
            units = chunker.chunk(content)

            all_entities: dict[tuple[str, str], Entity] = {}
            all_relations: list[Any] = []

            for unit in units:
                result = self._entity_extractor.extract_all(unit.text)
                for ext in result.get("entities", []):
                    key = (ext.text, ext.entity_type)
                    if key not in all_entities:
                        all_entities[key] = Entity(
                            id=None,
                            text=ext.text,
                            entity_type=ext.entity_type,
                            confidence=ext.confidence,
                            start_pos=ext.start_pos + unit.start,
                            end_pos=ext.end_pos + unit.start,
                            metadata=ext.metadata,
                        )
                all_relations.extend(result.get("relations", []))

            entities = list(all_entities.values())
            relations_count = len(all_relations)

            for entity in entities:
                self._store_entity_mention(entity, entry.id)

            self._store_relations(all_relations)

        return entry, entities, relations_count

    def _store_entity_mention(self, entity: Entity, memory_id: str | int) -> None:
        """Store an entity mention and link it to a memory.

        Args:
            entity: The extracted entity
            memory_id: The ID of the memory containing the entity
        """
        now = datetime.now()
        memory_id_str = str(memory_id)

        check_params = {
            "canonical_name": entity.text,
            "entity_type": entity.entity_type,
        }
        raw_result = self.conn.execute(
            """
            MATCH (e:Entity)
            WHERE e.canonical_name = $canonical_name
            AND e.entity_type = $entity_type
            RETURN e.id
            """,
            parameters=check_params,
        )
        result = _get_result(raw_result)

        entity_id: str
        if result.has_next():
            row = result.get_next()
            entity_id = str(row[0])
        else:
            embedding = self._get_embedding(entity.text)
            create_params = {
                "canonical_name": entity.text,
                "entity_type": entity.entity_type,
                "embedding": embedding,
                "created_at": now.strftime("%Y-%m-%d %H:%M:%S"),
                "updated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            }
            raw_result = self.conn.execute(
                """
                CREATE (e:Entity {
                    canonical_name: $canonical_name,
                    entity_type: $entity_type,
                    embedding: $embedding,
                    metadata: NULL,
                    created_at: timestamp($created_at),
                    updated_at: timestamp($updated_at)
                })
                RETURN e.id
                """,
                parameters=create_params,
            )
            result = _get_result(raw_result)
            row = result.get_next()
            entity_id = str(row[0])

        mention_params = {
            "entity_id": int(entity_id),
            "memory_id": int(memory_id_str),
            "mention_text": entity.text,
            "confidence": entity.confidence,
            "span_start": entity.start_pos,
            "span_end": entity.end_pos,
        }
        self.conn.execute(
            """
            MATCH (e:Entity), (m:Memory)
            WHERE e.id = $entity_id AND m.id = $memory_id
            CREATE (e)-[r:MentionedIn {
                mention_text: $mention_text,
                confidence: $confidence,
                span_start: $span_start,
                span_end: $span_end
            }]->(m)
            """,
            parameters=mention_params,
        )

    def _store_relations(self, relations: list) -> None:
        import json

        for rel in relations:
            source_embedding = self._get_embedding(rel.source_text)
            target_embedding = self._get_embedding(rel.target_text)

            source_result = self.conn.execute(
                """
                MATCH (e:Entity)
                RETURN e.id, e.canonical_name, e.embedding,
                       array_cosine_similarity(e.embedding, $query_embedding) AS score
                ORDER BY score DESC
                LIMIT 1
                """,
                parameters={"query_embedding": source_embedding},
            )
            res = _get_result(source_result)
            if not res.has_next():
                continue
            source_row = res.get_next()
            source_id = source_row[0]
            source_score = source_row[3] if len(source_row) > 3 else 0.0
            if source_score < 0.85:
                continue

            target_result = self.conn.execute(
                """
                MATCH (e:Entity)
                RETURN e.id, e.canonical_name, e.embedding,
                       array_cosine_similarity(e.embedding, $query_embedding) AS score
                ORDER BY score DESC
                LIMIT 1
                """,
                parameters={"query_embedding": target_embedding},
            )
            res = _get_result(target_result)
            if not res.has_next():
                continue
            target_row = res.get_next()
            target_id = target_row[0]
            target_score = target_row[3] if len(target_row) > 3 else 0.0
            if target_score < 0.85:
                continue

            rel_params = {
                "source_id": int(source_id),
                "target_id": int(target_id),
                "relation_type": rel.relation_type,
                "confidence": rel.confidence,
                "metadata": json.dumps(rel.metadata) if rel.metadata else None,
            }
            self.conn.execute(
                """
                MATCH (s:Entity), (t:Entity)
                WHERE s.id = $source_id AND t.id = $target_id
                CREATE (s)-[r:Relations {
                    relation_type: $relation_type,
                    confidence: $confidence,
                    metadata: $metadata
                }]->(t)
                """,
                parameters=rel_params,
            )

    def search_by_entity(
        self,
        entity_name: str,
        limit: int = 5,
    ) -> list[MemorySearchResult]:
        """Find memories mentioning a specific entity.

        Args:
            entity_name: The entity name to search for
            limit: Maximum number of results

        Returns:
            List of memories containing the entity
        """
        cypher = f"""
            MATCH (e:Entity)-[r:MentionedIn]->(m:Memory)
            WHERE e.canonical_name = '{entity_name.replace("'", "''")}'
            RETURN m.id, m.content, m.memory_type, m.importance, m.metadata,
                   m.created_at, m.updated_at, r.confidence
            ORDER BY r.confidence DESC, m.importance DESC
            LIMIT {limit}
        """

        raw_result = self.conn.execute(cypher)
        result = _get_result(raw_result)
        search_results = []

        while result.has_next():
            row = result.get_next()
            entry = self._row_to_entry(row)
            confidence = float(row[7]) if len(row) > 7 else 1.0
            search_results.append(MemorySearchResult(entry=entry, score=confidence))

        return search_results

    def get_entity_graph(
        self,
        entity_id: str,
        max_depth: int = 1,
    ) -> dict[str, Any]:
        """Get connected entities and their relationships.

        Args:
            entity_id: The entity ID to explore
            max_depth: Maximum relationship depth

        Returns:
            Dictionary with entity and related entities
        """
        # Get the main entity
        entity_cypher = f"""
            MATCH (e:Entity)
            WHERE e.id = {entity_id}
            RETURN e.id, e.canonical_name, e.entity_type, e.metadata
        """
        raw_result = self.conn.execute(entity_cypher)
        result = _get_result(raw_result)

        if not result.has_next():
            return {}

        row = result.get_next()
        entity_data = {
            "id": str(row[0]),
            "canonical_name": str(row[1]),
            "entity_type": str(row[2]),
            "metadata": row[3],
        }

        # Get related entities via coreference
        related_cypher = f"""
            MATCH (e:Entity)-[c:Coreference]->(related:Entity)
            WHERE e.id = {entity_id}
            RETURN related.id, related.canonical_name, related.entity_type, c.similarity_score
        """
        raw_result = self.conn.execute(related_cypher)
        result = _get_result(raw_result)

        related_entities = []
        while result.has_next():
            row = result.get_next()
            related_entities.append(
                {
                    "id": str(row[0]),
                    "canonical_name": str(row[1]),
                    "entity_type": str(row[2]),
                    "similarity_score": float(row[3]) if row[3] else 0.0,
                }
            )

        # Get mentioned memories
        memories_cypher = f"""
            MATCH (e:Entity)-[r:MentionedIn]->(m:Memory)
            WHERE e.id = {entity_id}
            RETURN m.id, m.content, r.confidence
            ORDER BY r.confidence DESC
            LIMIT 10
        """
        raw_result = self.conn.execute(memories_cypher)
        result = _get_result(raw_result)

        mentioned_memories = []
        while result.has_next():
            row = result.get_next()
            mentioned_memories.append(
                {
                    "memory_id": str(row[0]),
                    "content_preview": str(row[1])[:200] + "..."
                    if len(str(row[1])) > 200
                    else str(row[1]),
                    "mention_confidence": float(row[2]) if row[2] else 0.0,
                }
            )

        return {
            "entity": entity_data,
            "related_entities": related_entities,
            "mentioned_in": mentioned_memories,
        }

    def get_all_entities(
        self,
        entity_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get all entities, optionally filtered by type.

        Args:
            entity_type: Filter by entity type (optional)
            limit: Maximum number of results

        Returns:
            List of entity dictionaries
        """
        if entity_type:
            cypher = f"""
                MATCH (e:Entity)
                WHERE e.entity_type = '{entity_type}'
                RETURN e.id, e.canonical_name, e.entity_type, e.metadata
                LIMIT {limit}
            """
        else:
            cypher = f"""
                MATCH (e:Entity)
                RETURN e.id, e.canonical_name, e.entity_type, e.metadata
                LIMIT {limit}
            """

        raw_result = self.conn.execute(cypher)
        result = _get_result(raw_result)

        entities = []
        while result.has_next():
            row = result.get_next()
            entities.append(
                {
                    "id": str(row[0]),
                    "canonical_name": str(row[1]),
                    "entity_type": str(row[2]),
                    "metadata": row[3],
                }
            )

        return entities

    # Dynamic schema methods
    def create_dynamic_schema_tables(
        self,
        discovered_schemas: list[dict[str, Any]],
    ) -> dict[str, str]:
        """Create node and relationship tables for discovered schemas.

        Args:
            discovered_schemas: List of discovered schema types from clustering

        Returns:
            Dictionary mapping schema type names to table names
        """
        table_mapping = {}

        for schema in discovered_schemas:
            type_name = schema["type_name"]
            confidence = schema.get("confidence", 0.0)

            # Only create tables for high-confidence schemas
            if confidence < 0.5:
                continue

            # Sanitize type name for table name
            table_name = self._sanitize_table_name(type_name)

            # Check if table already exists
            try:
                # Create node table for this entity type
                self.conn.execute(
                    f"""
                    CREATE NODE TABLE IF NOT EXISTS {table_name}(
                        id SERIAL PRIMARY KEY,
                        canonical_name STRING,
                        entity_type STRING DEFAULT '{type_name}',
                        embedding FLOAT[384],
                        metadata JSON,
                        discovered_confidence FLOAT DEFAULT {confidence},
                        created_at TIMESTAMP,
                        updated_at TIMESTAMP
                    )
                    """
                )

                # Create relationship table for mentions
                rel_table_name = f"MentionedIn_{table_name}"
                self.conn.execute(
                    f"""
                    CREATE REL TABLE IF NOT EXISTS {rel_table_name}(
                        FROM {table_name} TO Memory,
                        mention_text STRING,
                        confidence FLOAT,
                        span_start INT64,
                        span_end INT64
                    )
                    """
                )

                table_mapping[type_name] = table_name

            except Exception as e:
                # Table might already exist or other error
                print(f"Warning: Could not create table for {type_name}: {e}")
                continue

        return table_mapping

    def populate_dynamic_schema_tables(
        self,
        entities: list[Entity],
        entity_to_type: dict[str, str],
        table_mapping: dict[str, str],
        memory_id: int | None = None,
    ) -> dict[str, int]:
        """Populate dynamic schema tables with entities from the generic Entity table.

        Args:
            entities: List of entities to populate
            entity_to_type: Mapping from entity ID to discovered type name
            table_mapping: Mapping from type name to table name
            memory_id: Optional memory ID to link entities to

        Returns:
            Dictionary with counts of entities stored per table
        """
        counts: dict[str, int] = {}

        for entity in entities:
            entity_type = entity_to_type.get(str(entity.id))
            if not entity_type:
                continue

            table_name = table_mapping.get(entity_type)
            if not table_name:
                continue

            # Store entity in typed table
            self._store_entity_in_typed_table(
                entity, str(memory_id) if memory_id else "0", table_name
            )
            counts[table_name] = counts.get(table_name, 0) + 1

        return counts

    def store_schema_hierarchy(
        self,
        discovered_schemas: list[dict[str, Any]],
    ) -> int:
        """Store discovered schema types in the hierarchy table.

        Args:
            discovered_schemas: List of discovered schema dicts with type_name, base_type, etc.

        Returns:
            Number of schema types stored
        """
        import json

        now = datetime.now()
        count = 0

        for schema in discovered_schemas:
            type_name = schema.get("type_name", "")
            base_type = schema.get("base_type", "unknown")
            confidence = schema.get("confidence", 0.0)
            sample_entities = schema.get("sample_entities", [])
            entity_count = schema.get("size", 0)

            if type_name.lower() == base_type.lower():
                continue

            check_params = {"type_name": type_name}
            raw_result = self.conn.execute(
                """
                MATCH (s:DiscoveredSchemaType)
                WHERE s.type_name = $type_name
                RETURN s.id
                """,
                parameters=check_params,
            )
            result = _get_result(raw_result)

            if result.has_next():
                continue

            samples_json = json.dumps(sample_entities[:10])
            create_params = {
                "type_name": type_name,
                "base_type": base_type,
                "confidence": confidence,
                "sample_entities": samples_json,
                "entity_count": entity_count,
                "created_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self.conn.execute(
                """
                CREATE (s:DiscoveredSchemaType {
                    type_name: $type_name,
                    base_type: $base_type,
                    confidence: $confidence,
                    sample_entities: CAST($sample_entities AS JSON),
                    entity_count: $entity_count,
                    created_at: timestamp($created_at)
                })
                """,
                parameters=create_params,
            )
            count += 1

        return count

    def get_schema_hierarchy(self) -> list[dict[str, Any]]:
        """Get all discovered schema types and their hierarchy.

        Returns:
            List of schema type dicts with type_name, base_type, etc.
        """
        import json

        cypher = """
            MATCH (s:DiscoveredSchemaType)
            RETURN s.type_name, s.base_type, s.confidence, s.sample_entities, s.entity_count
            ORDER BY s.base_type, s.confidence DESC
        """
        raw_result = self.conn.execute(cypher)
        result = _get_result(raw_result)

        schemas = []
        while result.has_next():
            row = result.get_next()
            samples_json = row[3] if row[3] else "[]"
            schemas.append(
                {
                    "type_name": row[0],
                    "base_type": row[1],
                    "confidence": row[2],
                    "sample_entities": json.loads(samples_json),
                    "entity_count": row[4],
                }
            )

        return schemas

    def _sanitize_table_name(self, name: str) -> str:
        """Sanitize a schema type name for use as a table name."""
        # Replace spaces and special characters with underscores
        import re

        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        # Ensure it starts with a letter
        if sanitized[0].isdigit():
            sanitized = "Entity_" + sanitized
        return sanitized

    def store_with_dynamic_schema(
        self,
        content: str,
        memory_type: str = "general",
        importance: int = 5,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[MemoryEntry, dict[str, Any]]:
        """Store memory and discover/create dynamic schema for entities.

        Args:
            content: The memory content
            memory_type: Type of memory
            importance: Importance score
            metadata: Additional metadata

        Returns:
            Tuple of (MemoryEntry, schema discovery results)
        """
        # Store the memory first
        entry = self.store(content, memory_type, importance, metadata)

        results = {
            "entities": [],
            "discovered_schemas": [],
            "table_mapping": {},
            "entity_to_type": {},
        }

        if not self._entity_extractor:
            return entry, results

        from memory.schema_discovery import DynamicSchemaDiscovery

        extractor = self._entity_extractor

        # Extract with broad labels to get entities
        entities = extractor.extract_with_context(
            content,
            context={"memory_id": entry.id, "memory_type": memory_type},
        )

        results["entities"] = entities

        if len(entities) == 0:
            return entry, results

        for entity in entities:
            entity.id = str(uuid.uuid4())

        schema_discovery = DynamicSchemaDiscovery(
            similarity_threshold=0.75,
            min_cluster_size=2,
            min_confidence=0.6,
        )

        discovered_schemas, entity_to_type = schema_discovery.discover_schema(entities)

        results["discovered_schemas"] = [
            {
                "type_name": s.type_name,
                "confidence": s.confidence,
                "sample_entities": s.sample_entities,
                "cluster_id": s.cluster_id,
                "size": s.size,
            }
            for s in discovered_schemas
        ]
        results["entity_to_type"] = entity_to_type

        if discovered_schemas:
            table_mapping = self.create_dynamic_schema_tables(
                results["discovered_schemas"]
            )
            results["table_mapping"] = table_mapping

            source_entity_ids: dict[str, int] = {}
            for entity in entities:
                self._store_entity_mention(entity, entry.id)
                check_params = {
                    "canonical_name": entity.text,
                    "entity_type": entity.entity_type,
                }
                raw_result = self.conn.execute(
                    """
                    MATCH (e:Entity)
                    WHERE e.canonical_name = $canonical_name
                    AND e.entity_type = $entity_type
                    RETURN e.id
                    """,
                    parameters=check_params,
                )
                res = _get_result(raw_result)
                if res.has_next():
                    row = res.get_next()
                    source_entity_ids[entity.id] = int(row[0])

            for entity in entities:
                if entity.id in entity_to_type:
                    entity_type = entity_to_type[entity.id]
                    if entity_type in table_mapping:
                        table_name = table_mapping[entity_type]
                        typed_entity_id = self._store_entity_in_typed_table(
                            entity, entry.id, table_name
                        )
                        source_id = source_entity_ids.get(entity.id)
                        if typed_entity_id is not None and source_id is not None:
                            self._create_sourced_from_relation(
                                typed_entity_id, source_id, table_name
                            )

        return entry, results

    def _store_entity_in_typed_table(
        self,
        entity: Entity,
        memory_id: str,
        table_name: str,
    ) -> int | None:
        """Store an entity in a type-specific table.

        Args:
            entity: The entity to store
            memory_id: ID of the memory mentioning the entity
            table_name: Name of the type-specific table

        Returns:
            The ID of the created/found entity, or None on failure
        """
        now = datetime.now()

        check_cypher = f"""
            MATCH (e:{table_name})
            WHERE e.canonical_name = '{entity.text.replace("'", "''")}'
            RETURN e.id
        """
        raw_result = self.conn.execute(check_cypher)
        result = _get_result(raw_result)

        entity_id: int
        if result.has_next():
            row = result.get_next()
            entity_id = int(row[0])
        else:
            embedding = self._get_embedding(entity.text)
            embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

            create_cypher = f"""
                CREATE (e:{table_name} {{
                    canonical_name: '{entity.text.replace("'", "''")}',
                    entity_type: '{entity.entity_type}',
                    embedding: {embedding_str},
                    metadata: NULL,
                    created_at: timestamp('{now.strftime("%Y-%m-%d %H:%M:%S")}'),
                    updated_at: timestamp('{now.strftime("%Y-%m-%d %H:%M:%S")}')
                }})
                RETURN e.id
            """
            raw_result = self.conn.execute(create_cypher)
            result = _get_result(raw_result)
            if not result.has_next():
                return None
            row = result.get_next()
            entity_id = int(row[0])

        rel_table_name = f"MentionedIn_{table_name}"
        mention_cypher = f"""
            MATCH (e:{table_name}), (m:Memory)
            WHERE e.id = {entity_id} AND m.id = {memory_id}
            CREATE (e)-[r:{rel_table_name} {{
                mention_text: '{entity.text.replace("'", "''")}',
                confidence: {entity.confidence},
                span_start: {entity.start_pos},
                span_end: {entity.end_pos}
            }}]->(m)
        """
        self.conn.execute(mention_cypher)

        return entity_id

    def _create_sourced_from_relation(
        self, typed_entity_id: int, source_entity_id: int, table_name: str
    ) -> None:
        """Create a SourcedFrom relation from typed entity to source Entity.

        Args:
            typed_entity_id: ID of the entity in the typed table
            source_entity_id: ID of the source entity in the generic Entity table
            table_name: Name of the typed table
        """
        rel_table_name = f"SourcedFrom_{table_name}"
        try:
            self.conn.execute(
                f"""
                CREATE REL TABLE IF NOT EXISTS {rel_table_name}(
                    FROM {table_name} TO Entity
                )
                """
            )
        except Exception:
            pass

        cypher = f"""
            MATCH (typed:{table_name}), (source:Entity)
            WHERE typed.id = {typed_entity_id} AND source.id = {source_entity_id}
            CREATE (typed)-[r:{rel_table_name}]->(source)
        """
        self.conn.execute(cypher)
