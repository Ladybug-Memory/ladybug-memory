from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class MemoryEntry:
    id: int
    content: str
    memory_type: str = "general"
    importance: int = 5
    metadata: dict[str, Any] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class MemorySearchResult:
    entry: MemoryEntry
    score: float = 1.0


class AgentMemory(ABC):
    @abstractmethod
    def store(
        self,
        content: str,
        memory_type: str = "general",
        importance: int = 5,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        limit: int = 5,
        memory_type: str | None = None,
    ) -> list[MemorySearchResult]:
        pass

    @abstractmethod
    def semantic_search(
        self,
        query: str,
        limit: int = 5,
        memory_type: str | None = None,
    ) -> list[MemorySearchResult]:
        pass

    @abstractmethod
    def recall(
        self,
        limit: int = 10,
        min_importance: int = 0,
        memory_type: str | None = None,
    ) -> list[MemoryEntry]:
        pass

    @abstractmethod
    def get(self, memory_id: str) -> MemoryEntry | None:
        pass

    @abstractmethod
    def update(
        self,
        memory_id: str,
        content: str | None = None,
        importance: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry | None:
        pass

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        pass

    @abstractmethod
    def link(
        self,
        source_id: str,
        target_id: str,
        relation: str = "related",
        start_timestamp: datetime | None = None,
        end_timestamp: datetime | None = None,
    ) -> bool:
        pass

    @abstractmethod
    def get_related(
        self,
        memory_id: str,
        relation: str | None = None,
        max_depth: int = 1,
    ) -> list[tuple[MemoryEntry, str]]:
        pass

    @abstractmethod
    def count(self) -> int:
        pass

    # Knowledge Graph methods
    @abstractmethod
    def extract_entities(
        self,
        content: str,
        labels: list[str] | None = None,
        threshold: float | None = None,
    ) -> list[Any]:
        """Extract entities from content using GLiNER2.

        Args:
            content: The text to extract entities from
            labels: Entity types to extract (uses defaults if None)
            threshold: Confidence threshold (uses default if None)

        Returns:
            List of extracted entities
        """
        pass

    @abstractmethod
    def search_by_entity(
        self,
        entity_name: str,
        limit: int = 5,
    ) -> list[Any]:
        """Find memories mentioning a specific entity.

        Args:
            entity_name: The entity name to search for
            limit: Maximum number of results

        Returns:
            List of memories containing the entity
        """
        pass

    @abstractmethod
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
        pass
