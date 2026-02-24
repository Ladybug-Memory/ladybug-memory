"""Logical document chunking for H-GLUE architecture."""

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class LogicalUnit:
    """A semantically coherent unit of text."""

    text: str
    unit_type: str
    start: int
    end: int
    metadata: dict[str, Any] | None = None


class LogicalChunker:
    """Chunk documents into logical units based on structure.

    H-GLUE architecture: uses semantic boundaries (paragraphs,
    sections) rather than fixed-size chunks to preserve context.
    """

    def __init__(
        self,
        min_unit_size: int = 100,
        max_unit_size: int = 2000,
        merge_threshold: int = 150,
    ):
        self.min_unit_size = min_unit_size
        self.max_unit_size = max_unit_size
        self.merge_threshold = merge_threshold

    def chunk(self, text: str) -> list[LogicalUnit]:
        """Split text into logical units."""
        units = []

        paragraphs = self._split_paragraphs(text)

        for para_text, para_start, para_end in paragraphs:
            if len(para_text.strip()) < self.min_unit_size:
                if units and len(units[-1].text) + len(para_text) < self.max_unit_size:
                    units[-1].text += "\n\n" + para_text
                    units[-1].end = para_end
                    continue

            if len(para_text) > self.max_unit_size:
                sub_units = self._split_sentence_groups(para_text, para_start)
                units.extend(sub_units)
            else:
                units.append(
                    LogicalUnit(
                        text=para_text,
                        unit_type="paragraph",
                        start=para_start,
                        end=para_end,
                    )
                )

        return self._finalize_units(units)

    def _split_paragraphs(self, text: str) -> list[tuple[str, int, int]]:
        """Split by blank lines into paragraphs."""
        paragraphs = []
        pattern = re.compile(r"\n\s*\n")
        last_end = 0

        for match in pattern.finditer(text):
            para_text = text[last_end : match.start()].strip()
            if para_text:
                paragraphs.append((para_text, last_end, match.start()))
            last_end = match.end()

        if last_end < len(text):
            para_text = text[last_end:].strip()
            if para_text:
                paragraphs.append((para_text, last_end, len(text)))

        return paragraphs

    def _split_sentence_groups(self, text: str, base_offset: int) -> list[LogicalUnit]:
        """Split long paragraph into sentence groups."""
        units = []
        sentences = re.split(r"(?<=[.!?])\s+", text)

        current = []
        current_start = base_offset
        current_len = 0

        for sent in sentences:
            if current_len + len(sent) > self.max_unit_size and current:
                unit_text = " ".join(current)
                units.append(
                    LogicalUnit(
                        text=unit_text,
                        unit_type="sentence_group",
                        start=current_start,
                        end=current_start + len(unit_text),
                    )
                )
                current = []
                current_start += current_len + 1
                current_len = 0

            current.append(sent)
            current_len += len(sent) + 1

        if current:
            unit_text = " ".join(current)
            units.append(
                LogicalUnit(
                    text=unit_text,
                    unit_type="sentence_group",
                    start=current_start,
                    end=current_start + len(unit_text),
                )
            )

        return units

    def _finalize_units(self, units: list[LogicalUnit]) -> list[LogicalUnit]:
        """Merge very small units with neighbors."""
        if len(units) <= 1:
            return units

        result = []
        i = 0
        while i < len(units):
            unit = units[i]

            if len(unit.text) < self.merge_threshold and result:
                result[-1].text += "\n\n" + unit.text
                result[-1].end = unit.end
            else:
                result.append(unit)
            i += 1

        return result
