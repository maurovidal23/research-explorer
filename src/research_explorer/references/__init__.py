"""Shared paper-level bibliography mapping and reference graph construction."""

from research_explorer.references.bibliography import (
    provisional_reference_id,
    raw_entry_hash,
    segment_bibliography,
    source_content_hash,
)
from research_explorer.references.builder import (
    ReferenceGraphBuilder,
    build_reference_builder,
)
from research_explorer.references.mapper import ReferenceMapper
from research_explorer.references.models import (
    EntryType,
    MappedEntry,
    MappingStatus,
    RawBibliographyEntry,
    ReferenceAccounting,
    ResolutionState,
)

__all__ = [
    "EntryType",
    "MappedEntry",
    "MappingStatus",
    "RawBibliographyEntry",
    "ReferenceAccounting",
    "ReferenceGraphBuilder",
    "ReferenceMapper",
    "ResolutionState",
    "build_reference_builder",
    "provisional_reference_id",
    "raw_entry_hash",
    "segment_bibliography",
    "source_content_hash",
]
