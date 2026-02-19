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
    ]

    def __init__(
        self,
        model_name: str = "fastino/gliner2-base-v1",
        confidence_threshold: float = 0.85,
        labels: list[str] | None = None,
        relations: list[str] | None = None,
    ):
        self.model = GLiNER2.from_pretrained(model_name)
        self.confidence_threshold = confidence_threshold
        self.labels = labels or self.DEFAULT_LABELS
        self.relations = relations or SCHEMA_ORG_RELATIONS

    def extract(
        self,
        text: str,
        labels: list[str] | None = None,
        threshold: float | None = None,
    ) -> list[ExtractedEntity]:
        use_labels = labels or self.labels
        use_threshold = (
            threshold if threshold is not None else self.confidence_threshold
        )

        schema = self.model.create_schema().entities(use_labels)
        result = self.model.extract(text, schema)

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
        relations: list[str] | None = None,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Extract entities, relations, and structured data in a single GLiNER2 call."""
        use_threshold = (
            threshold if threshold is not None else self.confidence_threshold
        )
        use_labels = labels or self.labels
        use_relations = relations or self.relations

        schema = (
            self.model.create_schema()
            .entities(use_labels)
            .relations(use_relations)
            .structure("money")
            .field("amount", dtype="str")
            .field("currency", dtype="str")
            .structure("number")
            .field("value", dtype="str")
            .field("unit", dtype="str")
            .structure("fraction")
            .field("value", dtype="str")
            .structure("percentage")
            .field("value", dtype="str")
        )

        result = self.model.extract(
            text, schema, threshold=use_threshold, include_confidence=True
        )

        entities: list[ExtractedEntity] = []
        for entity_type, entity_list in result.get("entities", {}).items():
            for entity_data in entity_list:
                if isinstance(entity_data, str):
                    entities.append(
                        ExtractedEntity(
                            text=entity_data,
                            entity_type=entity_type,
                            confidence=1.0,
                            start_pos=0,
                            end_pos=0,
                        )
                    )
                else:
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
        for rel_type, rel_list in result.get("relation_extraction", {}).items():
            for rel_data in rel_list:
                if isinstance(rel_data, tuple):
                    head_text, tail_text = rel_data
                    relations.append(
                        Relation(
                            id=None,
                            relation_type=rel_type,
                            source_text=head_text,
                            target_text=tail_text,
                            source_start=0,
                            source_end=0,
                            target_start=0,
                            target_end=0,
                            confidence=1.0,
                            metadata={
                                "extractor": "gliner2",
                                "relation_schema": "schema.org",
                            },
                        )
                    )
                elif isinstance(rel_data, dict):
                    head = rel_data.get("head", {})
                    tail = rel_data.get("tail", {})
                    confidence = rel_data.get("confidence", head.get("confidence", 1.0))
                    if confidence >= use_threshold:
                        relations.append(
                            Relation(
                                id=None,
                                relation_type=rel_type,
                                source_text=head.get("text", ""),
                                target_text=tail.get("text", ""),
                                source_start=head.get("start", 0),
                                source_end=head.get("end", 0),
                                target_start=tail.get("start", 0),
                                target_end=tail.get("end", 0),
                                confidence=confidence,
                                metadata={
                                    "extractor": "gliner2",
                                    "relation_schema": "schema.org",
                                },
                            )
                        )

        structured: dict[str, list[dict[str, Any]]] = {
            "money": [],
            "number": [],
            "fraction": [],
            "percentage": [],
        }

        for struct_type in ["money", "number", "fraction", "percentage"]:
            for item in result.get(struct_type, []):
                if isinstance(item, dict) and item:
                    structured[struct_type].append(item)

        return {
            "entities": entities,
            "relations": relations,
            "structured": structured,
        }


class AdaptiveEntityExtractor:
    """Two-tier extractor: GLiNER2 primary + optional LLM fallback."""

    def __init__(
        self,
        gliner_extractor: GLiNEREntityExtractor | None = None,
        llm_client: Any | None = None,
        enable_llm_fallback: bool = False,
    ):
        self.gliner = gliner_extractor or GLiNEREntityExtractor()
        self.llm_client = llm_client
        self.enable_llm_fallback = enable_llm_fallback

    def extract(
        self,
        text: str,
        context: dict[str, Any] | None = None,
    ) -> list[Entity]:
        gliner_entities = self.gliner.extract_with_context(text, context)

        if not self.enable_llm_fallback or not self.llm_client:
            return gliner_entities

        return gliner_entities

    def extract_all(
        self,
        text: str,
        labels: list[str] | None = None,
        relations: list[str] | None = None,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        return self.gliner.extract_all(text, labels, relations, threshold)
