from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class MemoryEntry:
    id: str
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
