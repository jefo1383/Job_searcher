"""LLM-based job match analysis module using Google GenAI SDK."""

from pathlib import Path
from typing import Any
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from config.settings import get_settings


class JobAnalysisResult(BaseModel):
    """Structured evaluation output returned by Gemini Flash.

    Attributes:
        score: Relevance score between 0 and 100.
        is_match: True if the offer matches candidate goals without dealbreakers.
        key_matches: Specific validated candidate skills matching the offer.
        missing_skills: Unmatched requirements or noticeable gaps.
        reasoning: Concise rationale justifying the assigned score.
    """

    score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Relevance match score between candidate and job (0-100).",
    )
    is_match: bool = Field(
        ...,
        description="Whether the job is a suitable target without dealbreakers.",
    )
    key_matches: list[str] = Field(
        default_factory=list,
        description="Skills and competencies aligning with the role.",
    )
    missing_skills: list[str] = Field(
        default_factory=list,
        description="Required competencies missing from candidate profile.",
    )
    reasoning: str = Field(
        ...,
        description="Brief summary explaining the rationale behind the score.",
    )


class JobAnalyzer:
    """Asynchronous analyzer evaluating job postings with Gemini Flash.

    Attributes:
        model_name: Gemini model identifier.
        profile_content: Candidate profile text loaded from Markdown.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        profile_path: Path | None = None,
    ) -> None:
        """Initialize the analyzer with settings and candidate profile.

        Args:
            api_key: Google AI Studio API key (defaults to settings value).
            model_name: Gemini model name (defaults to settings value).
            profile_path: Path to candidate profile Markdown file.

        Raises:
            FileNotFoundError: If the profile file does not exist.
        """
        settings = get_settings()

        resolved_key = (
            api_key if api_key is not None
            else settings.gemini_api_key.get_secret_value()
        )
        self._api_key = resolved_key
        self.model_name = model_name or settings.gemini_model

        target_profile_path = profile_path or settings.candidate_profile_path
        if not target_profile_path.exists():
            raise FileNotFoundError(
                f"Candidate profile not found at: {target_profile_path}"
            )

        # Chargement du profil candidat en mémoire
        self.profile_content = target_profile_path.read_text(encoding="utf-8")

        # Initialisation du client Google GenAI
        self.client = genai.Client(api_key=self._api_key or "DUMMY_KEY")

    def _build_system_instruction(self) -> str:
        """Construct system instructions embedding the candidate profile.

        Returns:
            str: System instruction text.
        """
        return (
            "You are an expert technical recruiter and automated job matching "
            "evaluator. Your objective is to assess the alignment between a "
            "scraped job posting and the following candidate profile.\n\n"
            f"=== CANDIDATE PROFILE & TARGETING DIRECTIVES ===\n"
            f"{self.profile_content}\n"
            f"================================================\n\n"
            "EVALUATION INSTRUCTIONS:\n"
            "1. Strictly enforce DEALBREAKERS:\n"
            "   - If the job is located on-site outside Bouches-du-Rhône (13) "
            "without full remote option -> is_match = False, score <= 30.\n"
            "   - If the role is predominantly Web Frontend / CMS / UI -> "
            "is_match = False, score <= 30.\n"
            "   - If the job requires a Senior profile or > 3 years of software "
            "engineering experience -> is_match = False, score <= 35.\n"
            "   - If > 50% of the primary required technologies are missing -> "
            "is_match = False, score <= 40.\n"
            "2. Score calculation (0 to 100):\n"
            "   - 80-100: Exceptional match with AI agents, Python, C systems, "
            "DevOps, or SOC analyst roles in target locations/remote.\n"
            "   - 65-79: Good match with minor skill gaps easily bridgeable.\n"
            "   - 40-64: Partial match or secondary interest.\n"
            "   - 0-39: Definite mismatch or hit by dealbreakers.\n"
            "3. Return strict structured JSON adhering to the provided schema."
        )

    async def analyze_job(
        self,
        title: str,
        company: str,
        description: str,
        platform: str = "",
    ) -> JobAnalysisResult:
        """Evaluate a job offer asynchronously against candidate profile.

        Args:
            title: Job title.
            company: Hiring company name.
            description: Full text description of the position.
            platform: Name of the origin platform.

        Returns:
            JobAnalysisResult: Validated Pydantic evaluation result.

        Raises:
            ValueError: If the Google AI Studio API key is not configured.
            RuntimeError: If the LLM call or response parsing fails.
        """
        if not self._api_key or self._api_key == "DUMMY_KEY":
            raise ValueError(
                "GEMINI_API_KEY is not set. Please provide a valid key in "
                ".env or pass it to JobAnalyzer."
            )

        # Construction du prompt d'évaluation de l'offre
        user_prompt = (
            f"JOB DETAILS TO EVALUATE:\n"
            f"- Platform: {platform or 'Unknown'}\n"
            f"- Company: {company}\n"
            f"- Title: {title}\n"
            f"- Description:\n{description}\n"
        )

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=JobAnalysisResult,
            temperature=0.1,
            system_instruction=self._build_system_instruction(),
        )

        try:
            # Appel asynchrone non-bloquant à l'API Gemini Flash
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=config,
            )

            # Traitement de l'objet parsé ou validation du texte brut JSON
            if hasattr(response, "parsed") and response.parsed is not None:
                parsed_result: Any = response.parsed
                if isinstance(parsed_result, JobAnalysisResult):
                    return parsed_result
                if isinstance(parsed_result, dict):
                    return JobAnalysisResult.model_validate(parsed_result)

            if response.text:
                return JobAnalysisResult.model_validate_json(response.text)

            raise RuntimeError("Gemini returned an empty response body.")

        except Exception as err:
            if isinstance(err, (ValueError, RuntimeError)):
                raise
            # Encapsulation des erreurs réseau ou de quota Google GenAI
            raise RuntimeError(
                f"Failed to evaluate job posting with Gemini: {err}"
            ) from err

    async def evaluate_and_decide_status(
        self,
        score_threshold: int,
        title: str,
        company: str,
        description: str,
        platform: str = "",
        max_missing_skills_percent: int = 50,
    ) -> tuple[JobAnalysisResult, bool]:
        """Analyze job and decide whether it qualifies for human review.

        Evaluates relevance score against score_threshold, and enforces the
        maximum percentage limit on missing skills.

        Args:
            score_threshold: Minimum score required for qualification.
            title: Job position title.
            company: Hiring organization name.
            description: Job description text.
            platform: Scraped platform source.
            max_missing_skills_percent: Max allowed ratio of missing skills.

        Returns:
            tuple[JobAnalysisResult, bool]: Result and qualification boolean.
        """
        result = await self.analyze_job(
            title=title,
            company=company,
            description=description,
            platform=platform,
        )

        # Vérification du score et de l'adéquation globale
        is_qualified = result.is_match and result.score >= score_threshold

        # Contrôle du ratio de compétences manquantes (ex: max 50%)
        total_skills = len(result.key_matches) + len(result.missing_skills)
        if total_skills > 0:
            missing_ratio = (len(result.missing_skills) / total_skills) * 100
            if missing_ratio > max_missing_skills_percent:
                is_qualified = False

        return result, is_qualified
