"""Core package for AI analysis and generator modules."""

from core.analyzer import JobAnalysisResult, JobAnalyzer
from core.generator import JobGenerator, TailoredApplication
from core.location_matcher import LocationMatcher
from core.question_solver import QuestionAnswerResult, ScreeningQuestionSolver

__all__ = [
    "JobAnalysisResult",
    "JobAnalyzer",
    "JobGenerator",
    "LocationMatcher",
    "QuestionAnswerResult",
    "ScreeningQuestionSolver",
    "TailoredApplication",
]
