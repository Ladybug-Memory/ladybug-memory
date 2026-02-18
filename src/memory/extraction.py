"""GLiNER2 entity extraction module using fastino-ai/gliner2.

Supports:
- Named entity extraction (person, organization, location, etc.)
- Document extraction (JSON with subfields: title, author, date, url)
- Common entities: money (currency, amount), number, fraction, percentage
- Relation extraction using schema.org relation types
"""

import re
import uuid
from typing import Any

from gliner2 import GLiNER2

from memory.entities import (
    DocumentEntity,
    Entity,
    ExtractedEntity,
    FractionEntity,
    MoneyEntity,
    NumberEntity,
    PercentageEntity,
    Relation,
)


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
    """Entity extractor using GLiNER2 model (fastino-ai/gliner2) with optional confidence-based routing."""

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

    COMMON_ENTITY_LABELS = ["money", "number", "fraction", "percentage"]

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

        # GLiNER2 returns {'entities': {'label': [{'text': ..., 'confidence': ...}]}}
        result = self.model.extract_entities(
            text,
            use_labels,
            include_confidence=True,
            include_spans=True,
        )

        entities = []
        for entity_type, entity_list in result.get("entities", {}).items():
            for entity_data in entity_list:
                # Handle both simple string and dict with metadata
                if isinstance(entity_data, str):
                    # Simple text-only extraction (shouldn't happen with include_confidence/spans)
                    entity = ExtractedEntity(
                        text=entity_data,
                        entity_type=entity_type,
                        confidence=1.0,
                        start_pos=0,
                        end_pos=0,
                    )
                    # Filter by threshold
                    if entity.confidence >= use_threshold:
                        entities.append(entity)
                else:
                    # Dict with text, confidence, start, end
                    confidence = entity_data.get("confidence", 0.0)
                    # Filter by threshold
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

    def extract_documents(
        self,
        text: str,
        threshold: float | None = None,
    ) -> list[DocumentEntity]:
        """Extract document entities with structured subfields.

        Args:
            text: The text to extract documents from
            threshold: Override default threshold

        Returns:
            List of extracted document entities with subfields
        """
        use_threshold = (
            threshold if threshold is not None else self.confidence_threshold
        )

        result = self.model.extract_entities(
            text,
            ["document"],
            include_confidence=True,
            include_spans=True,
        )

        documents = []
        for entity_data in result.get("entities", {}).get("document", []):
            if isinstance(entity_data, str):
                continue

            confidence = entity_data.get("confidence", 0.0)
            if confidence < use_threshold:
                continue

            doc_text = entity_data.get("text", "")
            doc = self._parse_document(
                doc_text,
                confidence,
                entity_data.get("start", 0),
                entity_data.get("end", 0),
            )
            documents.append(doc)

        return documents

    def _parse_document(
        self, text: str, confidence: float, start: int, end: int
    ) -> DocumentEntity:
        """Parse document text to extract subfields."""
        title = None
        author = None
        date = None
        url = None
        doc_type = None

        title_patterns = [
            r'["""]([^"""]+)["""]',
            r"[''']([^''']+)[''']",
            r"(?:titled?|called)\s+[:\"]?\s*([A-Z][^.]+?)(?:\.|$|,|by)",
            r"^([A-Z][^.!?]+[.!?])",
        ]
        for pattern in title_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                break

        author_patterns = [
            r"(?:by|author[s]?:?)\s+([A-Z][a-zA-Z\s]+?)(?:\.|,|$|on)",
            r"(?:written by|authored by)\s+([A-Z][a-zA-Z\s]+)",
        ]
        for pattern in author_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                author = match.group(1).strip()
                break

        date_patterns = [
            r"(?:published|released|dated?)\s+(on\s+)?([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})",
            r"(\d{4}[-/]\d{2}[-/]\d{2})",
            r"(\d{1,2}[-/]\d{1,2}[-/]\d{4})",
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date = (
                    match.group(2)
                    if match.lastindex and match.lastindex > 1
                    else match.group(1)
                )
                break

        url_match = re.search(
            r"https?://[^\s<>\"]+|www\.[^\s<>\"]+\.[a-z]{2,}", text, re.IGNORECASE
        )
        if url_match:
            url = url_match.group(0)

        doc_types = {
            "paper": "academic paper",
            "article": "article",
            "report": "report",
            "book": "book",
            "thesis": "thesis",
            "dissertation": "dissertation",
            "patent": "patent",
            "whitepaper": "whitepaper",
            "study": "study",
        }
        text_lower = text.lower()
        for keyword, dtype in doc_types.items():
            if keyword in text_lower:
                doc_type = dtype
                break

        return DocumentEntity(
            id=None,
            text=text,
            confidence=confidence,
            start_pos=start,
            end_pos=end,
            title=title,
            author=author,
            date=date,
            url=url,
            doc_type=doc_type,
        )

    def extract_money(
        self,
        text: str,
        threshold: float | None = None,
    ) -> list[MoneyEntity]:
        """Extract monetary values with currency and amount.

        Args:
            text: The text to extract money entities from
            threshold: Override default threshold

        Returns:
            List of extracted money entities
        """
        use_threshold = (
            threshold if threshold is not None else self.confidence_threshold
        )

        result = self.model.extract_entities(
            text,
            ["money"],
            include_confidence=True,
            include_spans=True,
        )

        money_entities = []
        for entity_data in result.get("entities", {}).get("money", []):
            if isinstance(entity_data, str):
                continue

            confidence = entity_data.get("confidence", 0.0)
            if confidence < use_threshold:
                continue

            money_text = entity_data.get("text", "")
            money = self._parse_money(
                money_text,
                confidence,
                entity_data.get("start", 0),
                entity_data.get("end", 0),
            )
            money_entities.append(money)

        return money_entities

    def _parse_money(
        self, text: str, confidence: float, start: int, end: int
    ) -> MoneyEntity:
        """Parse money text to extract currency and amount."""
        currency = None
        amount = None

        currency_symbols = {
            "$": "USD",
            "€": "EUR",
            "£": "GBP",
            "¥": "JPY",
            "₹": "INR",
            "₽": "RUB",
            "₩": "KRW",
            "C$": "CAD",
            "A$": "AUD",
            "NZ$": "NZD",
        }

        for symbol, code in currency_symbols.items():
            if symbol in text:
                currency = code
                break

        if not currency:
            currency_pattern = r"\b(USD|EUR|GBP|JPY|INR|CAD|AUD|NZD|CHF|CNY)\b"
            match = re.search(currency_pattern, text, re.IGNORECASE)
            if match:
                currency = match.group(1).upper()

        amount_pattern = r"[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|trillion|m|b|k))?"
        match = re.search(amount_pattern, text, re.IGNORECASE)
        if match:
            amount_str = match.group(0).lower().replace(",", "")
            multipliers = {
                "million": 1_000_000,
                "billion": 1_000_000_000,
                "trillion": 1_000_000_000_000,
                "m": 1_000_000,
                "b": 1_000_000_000,
                "k": 1_000,
            }
            for suffix, mult in multipliers.items():
                if suffix in amount_str:
                    amount_str = amount_str.replace(suffix, "").strip()
                    try:
                        amount = float(amount_str) * mult
                    except ValueError:
                        pass
                    break
            else:
                try:
                    amount = float(amount_str)
                except ValueError:
                    pass

        return MoneyEntity(
            id=None,
            text=text,
            confidence=confidence,
            start_pos=start,
            end_pos=end,
            amount=amount,
            currency=currency,
        )

    def extract_numbers(
        self,
        text: str,
        threshold: float | None = None,
    ) -> list[NumberEntity]:
        """Extract numeric values.

        Args:
            text: The text to extract number entities from
            threshold: Override default threshold

        Returns:
            List of extracted number entities
        """
        use_threshold = (
            threshold if threshold is not None else self.confidence_threshold
        )

        result = self.model.extract_entities(
            text,
            ["number"],
            include_confidence=True,
            include_spans=True,
        )

        numbers = []
        for entity_data in result.get("entities", {}).get("number", []):
            if isinstance(entity_data, str):
                continue

            confidence = entity_data.get("confidence", 0.0)
            if confidence < use_threshold:
                continue

            num_text = entity_data.get("text", "")
            num = self._parse_number(
                num_text,
                confidence,
                entity_data.get("start", 0),
                entity_data.get("end", 0),
            )
            numbers.append(num)

        return numbers

    def _parse_number(
        self, text: str, confidence: float, start: int, end: int
    ) -> NumberEntity:
        """Parse number text to extract value and unit."""
        value = None
        unit = None

        number_pattern = r"[-+]?[\d,]+(?:\.\d+)?(?:[eE][+-]?\d+)?"
        match = re.search(number_pattern, text)
        if match:
            try:
                value = float(match.group(0).replace(",", ""))
            except ValueError:
                pass

        units = [
            "kg",
            "g",
            "mg",
            "lb",
            "oz",
            "m",
            "cm",
            "mm",
            "km",
            "ft",
            "in",
            "miles",
            "seconds",
            "minutes",
            "hours",
            "days",
            "years",
            "%",
            "percent",
            "degrees",
            "°",
        ]
        text_lower = text.lower()
        for u in units:
            if u in text_lower:
                unit = u
                break

        return NumberEntity(
            id=None,
            text=text,
            confidence=confidence,
            start_pos=start,
            end_pos=end,
            value=value,
            unit=unit,
        )

    def extract_fractions(
        self,
        text: str,
        threshold: float | None = None,
    ) -> list[FractionEntity]:
        """Extract fractional values.

        Args:
            text: The text to extract fraction entities from
            threshold: Override default threshold

        Returns:
            List of extracted fraction entities
        """
        use_threshold = (
            threshold if threshold is not None else self.confidence_threshold
        )

        result = self.model.extract_entities(
            text,
            ["fraction"],
            include_confidence=True,
            include_spans=True,
        )

        fractions = []
        for entity_data in result.get("entities", {}).get("fraction", []):
            if isinstance(entity_data, str):
                continue

            confidence = entity_data.get("confidence", 0.0)
            if confidence < use_threshold:
                continue

            frac_text = entity_data.get("text", "")
            frac = self._parse_fraction(
                frac_text,
                confidence,
                entity_data.get("start", 0),
                entity_data.get("end", 0),
            )
            fractions.append(frac)

        return fractions

    def _parse_fraction(
        self, text: str, confidence: float, start: int, end: int
    ) -> FractionEntity:
        """Parse fraction text to extract numerator, denominator, and decimal value."""
        numerator = None
        denominator = None
        decimal_value = None

        frac_pattern = r"(\d+)\s*/\s*(\d+)"
        match = re.search(frac_pattern, text)
        if match:
            try:
                numerator = float(match.group(1))
                denominator = float(match.group(2))
                decimal_value = numerator / denominator
            except (ValueError, ZeroDivisionError):
                pass

        word_fracs = {
            "half": (1, 2),
            "quarter": (1, 4),
            "third": (1, 3),
            "two-thirds": (2, 3),
            "three-quarters": (3, 4),
            "one-half": (1, 2),
            "one-third": (1, 3),
            "one-quarter": (1, 4),
        }
        text_lower = text.lower().strip()
        if text_lower in word_fracs:
            numerator, denominator = word_fracs[text_lower]
            decimal_value = numerator / denominator

        return FractionEntity(
            id=None,
            text=text,
            confidence=confidence,
            start_pos=start,
            end_pos=end,
            numerator=numerator,
            denominator=denominator,
            decimal_value=decimal_value,
        )

    def extract_percentages(
        self,
        text: str,
        threshold: float | None = None,
    ) -> list[PercentageEntity]:
        """Extract percentage values.

        Args:
            text: The text to extract percentage entities from
            threshold: Override default threshold

        Returns:
            List of extracted percentage entities
        """
        use_threshold = (
            threshold if threshold is not None else self.confidence_threshold
        )

        result = self.model.extract_entities(
            text,
            ["percentage"],
            include_confidence=True,
            include_spans=True,
        )

        percentages = []
        for entity_data in result.get("entities", {}).get("percentage", []):
            if isinstance(entity_data, str):
                continue

            confidence = entity_data.get("confidence", 0.0)
            if confidence < use_threshold:
                continue

            pct_text = entity_data.get("text", "")
            pct = self._parse_percentage(
                pct_text,
                confidence,
                entity_data.get("start", 0),
                entity_data.get("end", 0),
            )
            percentages.append(pct)

        return percentages

    def _parse_percentage(
        self, text: str, confidence: float, start: int, end: int
    ) -> PercentageEntity:
        """Parse percentage text to extract value."""
        value = None

        pct_pattern = r"(\d+(?:\.\d+)?)\s*%?"
        match = re.search(pct_pattern, text)
        if match:
            try:
                value = float(match.group(1))
            except ValueError:
                pass

        return PercentageEntity(
            id=None,
            text=text,
            confidence=confidence,
            start_pos=start,
            end_pos=end,
            value=value,
        )

    def extract_relations(
        self,
        text: str,
        entity_labels: list[str] | None = None,
        relation_types: list[str] | None = None,
        threshold: float | None = None,
    ) -> list[Relation]:
        """Extract relations between entities using GLiNER2.

        Args:
            text: The text to extract relations from
            entity_labels: Entity types to consider for relations
            relation_types: Relation types to extract (uses SCHEMA_ORG_RELATIONS if None)
            threshold: Override default threshold

        Returns:
            List of extracted relations
        """
        use_threshold = (
            threshold if threshold is not None else self.confidence_threshold
        )
        use_labels = entity_labels or self.labels
        use_relations = relation_types or SCHEMA_ORG_RELATIONS

        entities = self.extract(text, use_labels, threshold)
        if len(entities) < 2:
            return []

        relations = []
        rel_labels = [f"{r}" for r in use_relations[:25]]

        try:
            result = self.model.extract_entities(
                text,
                rel_labels,
                include_confidence=True,
                include_spans=True,
            )

            for relation_type, entity_list in result.get("entities", {}).items():
                for entity_data in entity_list:
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
                                relation_type=relation_type,
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
        except Exception:
            pass

        return relations

    def _find_nearest_entity(
        self,
        entities: list[ExtractedEntity],
        position: int,
        before: bool = True,
    ) -> ExtractedEntity | None:
        """Find the nearest entity to a position.

        Args:
            entities: List of entities to search
            position: Position in text
            before: If True, find nearest entity before position; else after

        Returns:
            Nearest entity or None
        """
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

    def extract_all(
        self,
        text: str,
        labels: list[str] | None = None,
        threshold: float | None = None,
        include_documents: bool = True,
        include_money: bool = True,
        include_numbers: bool = True,
        include_fractions: bool = True,
        include_percentages: bool = True,
        include_relations: bool = False,
    ) -> dict[str, Any]:
        """Extract all entity types and relations from text in a single GLiNER2 call.

        Args:
            text: The text to extract from
            labels: Entity labels to extract
            threshold: Confidence threshold
            include_documents: Whether to extract documents
            include_money: Whether to extract money entities
            include_numbers: Whether to extract numbers
            include_fractions: Whether to extract fractions
            include_percentages: Whether to extract percentages
            include_relations: Whether to extract relations

        Returns:
            Dictionary with all extracted entities and relations
        """
        use_threshold = (
            threshold if threshold is not None else self.confidence_threshold
        )

        all_labels: list[str] = list(labels) if labels else list(self.labels)

        common_labels = []
        if include_documents:
            common_labels.append("document")
        if include_money:
            common_labels.append("money")
        if include_numbers:
            common_labels.append("number")
        if include_fractions:
            common_labels.append("fraction")
        if include_percentages:
            common_labels.append("percentage")

        for label in common_labels:
            if label not in all_labels:
                all_labels.append(label)

        relation_labels = []
        if include_relations:
            relation_labels = SCHEMA_ORG_RELATIONS[:25]
            for rel in relation_labels:
                if rel not in all_labels:
                    all_labels.append(rel)

        result = self.model.extract_entities(
            text,
            all_labels,
            include_confidence=True,
            include_spans=True,
        )

        entities_data = result.get("entities", {})

        base_entities: list[ExtractedEntity] = []
        entity_types = (
            set(labels) if labels else set(self.DEFAULT_LABELS) - set(common_labels)
        )
        entity_types.discard("document")
        entity_types.discard("money")
        entity_types.discard("number")
        entity_types.discard("fraction")
        entity_types.discard("percentage")

        for entity_type in entity_types:
            for entity_data in entities_data.get(entity_type, []):
                if isinstance(entity_data, str):
                    continue
                confidence = entity_data.get("confidence", 0.0)
                if confidence >= use_threshold:
                    base_entities.append(
                        ExtractedEntity(
                            text=entity_data.get("text", ""),
                            entity_type=entity_type,
                            confidence=confidence,
                            start_pos=entity_data.get("start", 0),
                            end_pos=entity_data.get("end", 0),
                        )
                    )

        output: dict[str, Any] = {"entities": base_entities}

        if include_documents:
            documents = []
            for entity_data in entities_data.get("document", []):
                if isinstance(entity_data, str):
                    continue
                confidence = entity_data.get("confidence", 0.0)
                if confidence >= use_threshold:
                    documents.append(
                        self._parse_document(
                            entity_data.get("text", ""),
                            confidence,
                            entity_data.get("start", 0),
                            entity_data.get("end", 0),
                        )
                    )
            output["documents"] = documents

        if include_money:
            money_entities = []
            for entity_data in entities_data.get("money", []):
                if isinstance(entity_data, str):
                    continue
                confidence = entity_data.get("confidence", 0.0)
                if confidence >= use_threshold:
                    money_entities.append(
                        self._parse_money(
                            entity_data.get("text", ""),
                            confidence,
                            entity_data.get("start", 0),
                            entity_data.get("end", 0),
                        )
                    )
            output["money"] = money_entities

        if include_numbers:
            numbers = []
            for entity_data in entities_data.get("number", []):
                if isinstance(entity_data, str):
                    continue
                confidence = entity_data.get("confidence", 0.0)
                if confidence >= use_threshold:
                    numbers.append(
                        self._parse_number(
                            entity_data.get("text", ""),
                            confidence,
                            entity_data.get("start", 0),
                            entity_data.get("end", 0),
                        )
                    )
            output["numbers"] = numbers

        if include_fractions:
            fractions = []
            for entity_data in entities_data.get("fraction", []):
                if isinstance(entity_data, str):
                    continue
                confidence = entity_data.get("confidence", 0.0)
                if confidence >= use_threshold:
                    fractions.append(
                        self._parse_fraction(
                            entity_data.get("text", ""),
                            confidence,
                            entity_data.get("start", 0),
                            entity_data.get("end", 0),
                        )
                    )
            output["fractions"] = fractions

        if include_percentages:
            percentages = []
            for entity_data in entities_data.get("percentage", []):
                if isinstance(entity_data, str):
                    continue
                confidence = entity_data.get("confidence", 0.0)
                if confidence >= use_threshold:
                    percentages.append(
                        self._parse_percentage(
                            entity_data.get("text", ""),
                            confidence,
                            entity_data.get("start", 0),
                            entity_data.get("end", 0),
                        )
                    )
            output["percentages"] = percentages

        if include_relations and len(base_entities) >= 2:
            relations = []
            for rel_type in relation_labels:
                for entity_data in entities_data.get(rel_type, []):
                    if isinstance(entity_data, str):
                        continue
                    confidence = entity_data.get("confidence", 0.0)
                    if confidence < use_threshold:
                        continue

                    rel_start = entity_data.get("start", 0)
                    rel_end = entity_data.get("end", 0)

                    source_entity = self._find_nearest_entity(
                        base_entities, rel_start, before=True
                    )
                    target_entity = self._find_nearest_entity(
                        base_entities, rel_end, before=False
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
            output["relations"] = relations

        return output

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
        result = self.model.extract_entities(
            text,
            use_labels,
            include_confidence=True,
            include_spans=True,
        )

        low_confidence = []
        for entity_type, entity_list in result.get("entities", {}).items():
            for entity_data in entity_list:
                if isinstance(entity_data, dict):
                    confidence = entity_data.get("confidence", 0.0)
                    if confidence < self.confidence_threshold:
                        low_confidence.append(
                            (
                                entity_data.get("start", 0),
                                entity_data.get("end", 0),
                                confidence,
                            )
                        )

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

    def extract_all(
        self,
        text: str,
        context: dict[str, Any] | None = None,
        include_documents: bool = True,
        include_money: bool = True,
        include_numbers: bool = True,
        include_fractions: bool = True,
        include_percentages: bool = True,
        include_relations: bool = False,
    ) -> dict[str, Any]:
        """Extract all entity types and relations using GLiNER2 (single call).

        Args:
            text: The text to extract from
            context: Additional context for extraction
            include_documents: Whether to extract documents
            include_money: Whether to extract money entities
            include_numbers: Whether to extract numbers
            include_fractions: Whether to extract fractions
            include_percentages: Whether to extract percentages
            include_relations: Whether to extract relations

        Returns:
            Dictionary with all extracted entities and relations
        """
        return self.gliner.extract_all(
            text,
            labels=None,
            threshold=None,
            include_documents=include_documents,
            include_money=include_money,
            include_numbers=include_numbers,
            include_fractions=include_fractions,
            include_percentages=include_percentages,
            include_relations=include_relations,
        )
