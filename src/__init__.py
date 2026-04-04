"""
Analysis module — cross-company comparison, temporal analysis,
multi-document QA, and filing change detection.
"""

from src.analysis.multi_doc_qa import MultiDocQA, run_multi_doc_query
from src.analysis.change_detection import FilingChangeDetector, run_change_detection

__all__ = [
    "MultiDocQA",
    "run_multi_doc_query",
    "FilingChangeDetector",
    "run_change_detection",
]