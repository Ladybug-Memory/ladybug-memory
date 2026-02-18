"""Entity dataclasses for knowledge graph extraction."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class Entity:
    """Represents an extracted entity mention."""

    id: str | None
    text: str
    entity_type: str
    confidence: float
    start_pos: int
    end_pos: int
    metadata: dict[str, Any] | None = None


@dataclass
class DocumentEntity:
    """Represents an extracted document with structured subfields."""

    id: str | None
    text: str
    confidence: float
    start_pos: int
    end_pos: int
    title: str | None = None
    author: str | None = None
    date: str | None = None
    url: str | None = None
    doc_type: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class MoneyEntity:
    """Represents a monetary value with currency and amount."""

    id: str | None
    text: str
    confidence: float
    start_pos: int
    end_pos: int
    amount: float | None = None
    currency: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class NumberEntity:
    """Represents a numeric value."""

    id: str | None
    text: str
    confidence: float
    start_pos: int
    end_pos: int
    value: float | None = None
    unit: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class FractionEntity:
    """Represents a fractional value."""

    id: str | None
    text: str
    confidence: float
    start_pos: int
    end_pos: int
    numerator: float | None = None
    denominator: float | None = None
    decimal_value: float | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class PercentageEntity:
    """Represents a percentage value."""

    id: str | None
    text: str
    confidence: float
    start_pos: int
    end_pos: int
    value: float | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class Relation:
    """Represents an extracted relation between entities."""

    id: str | None
    relation_type: str
    source_text: str
    target_text: str
    source_start: int
    source_end: int
    target_start: int
    target_end: int
    confidence: float
    metadata: dict[str, Any] | None = None


@dataclass
class EntityMention:
    """Represents a specific mention of an entity in content."""

    entity_id: int
    memory_id: int
    text: str
    start_pos: int
    end_pos: int
    confidence: float
    context: str | None = None


@dataclass
class CanonicalEntity:
    """Represents a canonical entity after disambiguation."""

    id: int
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
