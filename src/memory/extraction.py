"""GLiNER2 entity extraction module using fastino-ai/gliner2."""

import uuid
from typing import Any

from gliner2 import GLiNER2

from memory.entities import Entity, ExtractedEntity, Relation


SCHEMA_ORG_RELATIONS = [
    "author",
    "publisher",
    "about",
    "mentions",
    "location",
    "founder",
    "employee",
    "member",
    "parentOrganization",
    "subOrganization",
    "brand",
    "knows",
    "worksFor",
    "affiliatedWith",
    "spouse",
    "parent",
    "children",
    "sibling",
    "colleague",
    "contributor",
    "sponsor",
    "investor",
    "customer",
    "supplier",
    "competitor",
]


class GLiNEREntityExtractor:
    """Entity extractor using GLiNER2 model (fastino-ai/gliner2)."""

    DEFAULT_LABELS = [
        "person",
        "organization",
        "location",
        "product",
        "event",
        "date",
        "technology",
        "concept",
        "money",
        "number",
        "fraction",
        "percentage",
        "document",
    ]

    def __init__(
        self,
        model_name: str = "fastino/gliner2-base-v1",
        confidence_threshold: float = 0.85,
        labels: list[str] | None = None,
    ):
        """Initialize the GLiNER2 extractor.

        Args:
            model_name: HuggingFace model name for GLiNER2
            confidence_threshold: Minimum confidence for entity acceptance
            labels: List of entity labels to extract (uses defaults if None)
        """
        self.model = GLiNER2.from_pretrained(model_name)
        self.confidence_threshold = confidence_threshold
        self.labels = labels or self.DEFAULT_LABELS

    def extract(
        self,
        text: str,
        labels: list[str] | None = None,
        threshold: float | None = None,
    ) -> list[ExtractedEntity]:
        """Extract entities from text.

        Args:
            text: The text to extract entities from
            labels: Override default labels for this extraction
            threshold: Override default threshold for this extraction

        Returns:
            List of extracted entities
        """
        use_labels = labels or self.labels
        use_threshold = (
            threshold if threshold is not None else self.confidence_threshold
        )

        result = self.model.extract_entities(
            text,
            use_labels,
            include_confidence=True,
            include_spans=True,
        )

        entities = []
        for entity_type, entity_list in result.get("entities", {}).items():
            for entity_data in entity_list:
                if isinstance(entity_data, str):
                    entity = ExtractedEntity(
                        text=entity_data,
                        entity_type=entity_type,
                        confidence=1.0,
                        start_pos=0,
                        end_pos=0,
                    )
                    if entity.confidence >= use_threshold:
                        entities.append(entity)
                else:
                    confidence = entity_data.get("confidence", 0.0)
                    if confidence >= use_threshold:
                        entity = ExtractedEntity(
                            text=entity_data.get("text", ""),
                            entity_type=entity_type,
                            confidence=confidence,
                            start_pos=entity_data.get("start", 0),
                            end_pos=entity_data.get("end", 0),
                        )
                        entities.append(entity)

        return entities

    def extract_with_context(
        self,
        text: str,
        context: dict[str, Any] | None = None,
        labels: list[str] | None = None,
        threshold: float | None = None,
    ) -> list[Entity]:
        """Extract entities with context metadata.

        Args:
            text: The text to extract entities from
            context: Additional context (e.g., document_title, section_title)
            labels: Override default labels
            threshold: Override default threshold

        Returns:
            List of entities with unique IDs and metadata
        """
        extracted = self.extract(text, labels, threshold)

        entities = []
        for ext in extracted:
            metadata = context.copy() if context else {}
            metadata.update(
                {
                    "extractor": "gliner2",
                    "model": "fastino/gliner2",
                }
            )

            entity = Entity(
                id=str(uuid.uuid4()),
                text=ext.text,
                entity_type=ext.entity_type,
                confidence=ext.confidence,
                start_pos=ext.start_pos,
                end_pos=ext.end_pos,
                metadata=metadata,
            )
            entities.append(entity)

        return entities

    def extract_all(
        self,
        text: str,
        labels: list[str] | None = None,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Extract all entities and relations from text in a single GLiNER2 call.

        Args:
            text: The text to extract from
            labels: Entity labels to extract (adds schema.org relations automatically)
            threshold: Confidence threshold

        Returns:
            Dictionary with 'entities' and 'relations' keys
        """
        use_threshold = (
            threshold if threshold is not None else self.confidence_threshold
        )
        use_labels = list(labels) if labels else list(self.labels)

        for rel in SCHEMA_ORG_RELATIONS:
            if rel not in use_labels:
                use_labels.append(rel)

        result = self.model.extract_entities(
            text,
            use_labels,
            include_confidence=True,
            include_spans=True,
        )

        entities_data = result.get("entities", {})

        entity_types = set(labels) if labels else set(self.labels)
        entity_types = entity_types - set(SCHEMA_ORG_RELATIONS)

        entities: list[ExtractedEntity] = []
        for entity_type in entity_types:
            for entity_data in entities_data.get(entity_type, []):
                if isinstance(entity_data, str):
                    continue
                confidence = entity_data.get("confidence", 0.0)
                if confidence >= use_threshold:
                    entities.append(
                        ExtractedEntity(
                            text=entity_data.get("text", ""),
                            entity_type=entity_type,
                            confidence=confidence,
                            start_pos=entity_data.get("start", 0),
                            end_pos=entity_data.get("end", 0),
                        )
                    )

        relations: list[Relation] = []
        if len(entities) >= 2:
            for rel_type in SCHEMA_ORG_RELATIONS:
                for entity_data in entities_data.get(rel_type, []):
                    if isinstance(entity_data, str):
                        continue
                    confidence = entity_data.get("confidence", 0.0)
                    if confidence < use_threshold:
                        continue

                    rel_start = entity_data.get("start", 0)
                    rel_end = entity_data.get("end", 0)

                    source_entity = self._find_nearest_entity(
                        entities, rel_start, before=True
                    )
                    target_entity = self._find_nearest_entity(
                        entities, rel_end, before=False
                    )

                    if source_entity and target_entity:
                        relations.append(
                            Relation(
                                id=None,
                                relation_type=rel_type,
                                source_text=source_entity.text,
                                target_text=target_entity.text,
                                source_start=source_entity.start_pos,
                                source_end=source_entity.end_pos,
                                target_start=target_entity.start_pos,
                                target_end=target_entity.end_pos,
                                confidence=confidence,
                                metadata={
                                    "extractor": "gliner2",
                                    "relation_schema": "schema.org",
                                },
                            )
                        )

        return {"entities": entities, "relations": relations}

    def _find_nearest_entity(
        self,
        entities: list[ExtractedEntity],
        position: int,
        before: bool = True,
    ) -> ExtractedEntity | None:
        """Find the nearest entity to a position."""
        nearest = None
        min_distance = float("inf")

        for entity in entities:
            if before:
                if entity.end_pos <= position:
                    distance = position - entity.end_pos
                    if distance < min_distance:
                        min_distance = distance
                        nearest = entity
            else:
                if entity.start_pos >= position:
                    distance = entity.start_pos - position
                    if distance < min_distance:
                        min_distance = distance
                        nearest = entity

        return nearest


class AdaptiveEntityExtractor:
    """Two-tier extractor: GLiNER2 primary + optional LLM fallback for complex cases."""

    def __init__(
        self,
        gliner_extractor: GLiNEREntityExtractor | None = None,
        llm_client: Any | None = None,
        enable_llm_fallback: bool = False,
    ):
        """Initialize adaptive extractor.

        Args:
            gliner_extractor: GLiNER2 extractor instance (creates default if None)
            llm_client: LLM client for fallback extraction (e.g., OpenAI client)
            enable_llm_fallback: Whether to use LLM for low-confidence extractions
        """
        self.gliner = gliner_extractor or GLiNEREntityExtractor()
        self.llm_client = llm_client
        self.enable_llm_fallback = enable_llm_fallback

    def extract(
        self,
        text: str,
        context: dict[str, Any] | None = None,
    ) -> list[Entity]:
        """Extract entities using GLiNER2 with optional LLM fallback."""
        gliner_entities = self.gliner.extract_with_context(text, context)

        if not self.enable_llm_fallback or not self.llm_client:
            return gliner_entities

        return gliner_entities

    def extract_all(
        self,
        text: str,
        labels: list[str] | None = None,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Extract all entity types and relations using GLiNER2."""
        return self.gliner.extract_all(text, labels, threshold)
