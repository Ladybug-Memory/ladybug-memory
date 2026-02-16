"""GLiNER2 entity extraction module."""

import uuid
from typing import Any

from gliner import GLiNER

from memory.entities import Entity, ExtractedEntity


class GLiNEREntityExtractor:
    """Entity extractor using GLiNER2 model with optional confidence-based routing."""

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
        model_name: str = "urchade/gliner_medium-v2.1",
        confidence_threshold: float = 0.85,
        labels: list[str] | None = None,
    ):
        """Initialize the GLiNER2 extractor.

        Args:
            model_name: HuggingFace model name for GLiNER
            confidence_threshold: Minimum confidence for entity acceptance
            labels: List of entity labels to extract (uses defaults if None)
        """
        self.model = GLiNER.from_pretrained(model_name)
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

        predictions = self.model.predict_entities(
            text, use_labels, threshold=use_threshold
        )

        entities = []
        for pred in predictions:
            entity = ExtractedEntity(
                text=pred["text"],
                entity_type=pred["label"],
                confidence=pred["score"],
                start_pos=pred["start"],
                end_pos=pred["end"],
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
                    "extractor": "gliner",
                    "model": self.model.config.name_or_path
                    if hasattr(self.model, "config")
                    else "gliner",
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

    def extract_low_confidence_spans(
        self,
        text: str,
        labels: list[str] | None = None,
    ) -> list[tuple[int, int, float]]:
        """Extract entity spans with confidence below threshold.

        Useful for routing to LLM fallback for complex cases.

        Args:
            text: The text to analyze
            labels: Override default labels

        Returns:
            List of (start, end, confidence) tuples for low-confidence spans
        """
        use_labels = labels or self.labels
        predictions = self.model.predict_entities(text, use_labels, threshold=0.0)

        low_confidence = []
        for pred in predictions:
            if pred["score"] < self.confidence_threshold:
                low_confidence.append((pred["start"], pred["end"], pred["score"]))

        return low_confidence


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
        """Extract entities using GLiNER2 with optional LLM fallback.

        Args:
            text: The text to extract entities from
            context: Additional context for extraction

        Returns:
            List of entities (from GLiNER2 or combined with LLM fallback)
        """
        # Try GLiNER2 first
        gliner_entities = self.gliner.extract_with_context(text, context)

        if not self.enable_llm_fallback or not self.llm_client:
            return gliner_entities

        # Get low-confidence spans for LLM fallback
        low_conf_spans = self.gliner.extract_low_confidence_spans(text)

        if not low_conf_spans:
            return gliner_entities

        # Use LLM for low-confidence regions
        llm_entities = self._extract_with_llm(text, low_conf_spans, context)

        # Merge results (deduplicate overlapping spans)
        return self._merge_extractions(gliner_entities, llm_entities)

    def _extract_with_llm(
        self,
        text: str,
        spans: list[tuple[int, int, float]],
        context: dict[str, Any] | None,
    ) -> list[Entity]:
        """Extract entities using LLM for low-confidence spans.

        Args:
            text: Full text
            spans: Low-confidence spans (start, end, confidence)
            context: Additional context

        Returns:
            List of entities extracted by LLM
        """
        # Extract surrounding context for each span
        regions = []
        for start, end, _ in spans:
            context_start = max(0, start - 50)
            context_end = min(len(text), end + 50)
            regions.append(text[context_start:context_end])

        # This is a placeholder - implement actual LLM call
        # TODO: Implement LLM-based entity extraction
        return []

    def _merge_extractions(
        self,
        gliner_entities: list[Entity],
        llm_entities: list[Entity],
    ) -> list[Entity]:
        """Merge GLiNER2 and LLM extractions, removing duplicates.

        Args:
            gliner_entities: Entities from GLiNER2
            llm_entities: Entities from LLM

        Returns:
            Merged list without overlapping spans
        """
        all_entities = gliner_entities + llm_entities

        # Sort by position
        all_entities.sort(key=lambda e: (e.start_pos, -e.confidence))

        # Remove overlapping spans (keep higher confidence)
        merged = []
        for entity in all_entities:
            overlap = False
            for existing in merged:
                # Check for overlap
                if not (
                    entity.end_pos <= existing.start_pos
                    or entity.start_pos >= existing.end_pos
                ):
                    overlap = True
                    break
            if not overlap:
                merged.append(entity)

        return merged
