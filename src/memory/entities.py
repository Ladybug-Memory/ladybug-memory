"""Entity dataclasses for knowledge graph extraction."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class Entity:
    """Represents an extracted entity mention."""

    id: str
    text: str
    entity_type: str
    confidence: float
    start_pos: int
    end_pos: int
    metadata: dict[str, Any] | None = None


@dataclass
class EntityMention:
    """Represents a specific mention of an entity in content."""

    entity_id: str
    memory_id: str
    text: str
    start_pos: int
    end_pos: int
    confidence: float
    context: str | None = None


@dataclass
class CanonicalEntity:
    """Represents a canonical entity after disambiguation."""

    id: str
    canonical_name: str
    entity_type: str
    aliases: list[str]
    embedding: list[float] | None = None
    metadata: dict[str, Any] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class EntityCluster:
    """A cluster of entity mentions representing the same real-world entity."""

    entities: list[Entity]
    canonical_name: str
    similarity_score: float


@dataclass
class ExtractedEntity:
    """Result of entity extraction from content."""

    text: str
    entity_type: str
    confidence: float
    start_pos: int
    end_pos: int
    metadata: dict[str, Any] | None = None
