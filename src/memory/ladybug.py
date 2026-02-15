import uuid
from datetime import datetime
from typing import Any, cast

import real_ladybug as lb

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
    def __init__(self, db_path: str):
        self.db = lb.Database(db_path)
        self.conn = lb.Connection(self.db)
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute("INSTALL JSON; LOAD EXTENSION JSON;")
        self.conn.execute(
            """
            CREATE NODE TABLE IF NOT EXISTS Memory(
                id STRING PRIMARY KEY,
                content STRING,
                memory_type STRING,
                importance INT64,
                metadata JSON,
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

    def _row_to_entry(self, row: list | dict) -> MemoryEntry:
        if isinstance(row, dict):
            return MemoryEntry(
                id=cast(str, row.get("id")),
                content=cast(str, row.get("content")),
                memory_type=cast(str, row.get("memory_type")),
                importance=cast(int, row.get("importance")),
                metadata=row.get("metadata"),
                created_at=cast(datetime, row.get("created_at")),
                updated_at=cast(datetime, row.get("updated_at")),
            )
        return MemoryEntry(
            id=str(row[0]),
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
        memory_id = str(uuid.uuid4())
        now = datetime.now()
        metadata_json = f"CAST('{str(metadata)}' AS JSON)" if metadata else "NULL"

        self.conn.execute(
            f"""
            CREATE (m:Memory {{
                id: '{memory_id}',
                content: '{content.replace("'", "''")}',
                memory_type: '{memory_type}',
                importance: {importance},
                metadata: {metadata_json},
                created_at: timestamp('{now.strftime("%Y-%m-%d %H:%M:%S")}'),
                updated_at: timestamp('{now.strftime("%Y-%m-%d %H:%M:%S")}')
            }})
            RETURN m.id
            """
        )

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
                MATCH (m:Memory)
                WHERE m.memory_type = '{memory_type}'
                AND m.content CONTAINS '{query.replace("'", "''")}'
                RETURN m.id, m.content, m.memory_type, m.importance, m.metadata, m.created_at, m.updated_at
                ORDER BY m.importance DESC
                LIMIT {limit}
            """
        else:
            cypher = f"""
                MATCH (m:Memory)
                WHERE m.content CONTAINS '{query.replace("'", "''")}'
                RETURN m.id, m.content, m.memory_type, m.importance, m.metadata, m.created_at, m.updated_at
                ORDER BY m.importance DESC
                LIMIT {limit}
            """

        raw_result = self.conn.execute(cypher)
        result = _get_result(raw_result)
        search_results = []

        while result.has_next():
            row = result.get_next()
            entry = self._row_to_entry(row)
            score = 1.0 if query.lower() in entry.content.lower() else 0.5
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
            WHERE m.id = '{memory_id}'
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
            updates.append(f"m.content = '{content.replace("'", "''")}'")
        if importance is not None:
            updates.append(f"m.importance = {importance}")
        if metadata is not None:
            updates.append(f"m.metadata = CAST('{str(metadata)}' AS JSON)")

        if not updates:
            return self.get(memory_id)

        now = datetime.now()
        updates.append(f"m.updated_at = timestamp('{now.strftime("%Y-%m-%d %H:%M:%S")}')")

        cypher = f"""
            MATCH (m:Memory)
            WHERE m.id = '{memory_id}'
            SET {', '.join(updates)}
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
            WHERE m.id = '{memory_id}'
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
            WHERE a.id = '{source_id}' AND b.id = '{target_id}'
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
                WHERE m.id = '{memory_id}'
                RETURN related.id, related.content, related.memory_type, related.importance, related.metadata, related.created_at, related.updated_at, r.relation
            """
        else:
            cypher = f"""
                MATCH (m:Memory)-[r:MemoryLink]->(related:Memory)
                WHERE m.id = '{memory_id}'
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
