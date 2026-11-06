"""Core package for AI analysis and generator modules."""

from core.analyzer import JobAnalysisResult, JobAnalyzer
from core.generator import JobGenerator, TailoredApplication

__all__ = [
    "JobAnalysisResult",
    "JobAnalyzer",
    "JobGenerator",
    "TailoredApplication",
]
