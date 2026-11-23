"""LLM-based job match analysis module using Google GenAI SDK."""

import asyncio
from pathlib import Path
from typing import Any
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from config.criteria import parse_search_criteria_file
from config.settings import get_settings
from core.circuit_breaker import get_circuit_breaker


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
    company: str | None = Field(
        default=None,
        description="Actual hiring company name extracted from job text "
        "if provided name is generic, missing, or portal-based.",
    )
    contract_type: str | None = Field(
        default=None,
        description="Employment contract category extracted from offer: "
        "'Alternance', 'Stage', 'CDI', 'CDD', 'Intérim', 'Freelance', or null.",
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
        allowed_contracts: list[str] | None = None,
    ) -> None:
        """Initialize the analyzer with settings and candidate profile.

        Args:
            api_key: Google AI Studio API key (defaults to settings value).
            model_name: Gemini model name (defaults to settings value).
            profile_path: Path to candidate profile Markdown file.
            allowed_contracts: Optional list of acceptable contract types.

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

        # Configuration des types de contrats autorisés (par défaut depuis critères)
        if allowed_contracts is not None:
            self.allowed_contracts = [
                c.strip().lower() for c in allowed_contracts if c.strip()
            ]
        else:
            try:
                criteria = parse_search_criteria_file(
                    settings.search_criteria_path
                )
                self.allowed_contracts = [
                    c.strip().lower() for c in criteria.contracts if c.strip()
                ]
            except Exception:
                self.allowed_contracts = [
                    "alternance",
                    "stage",
                    "cdi",
                    "cdd",
                    "interim",
                ]

        # Initialisation du client Google GenAI
        self.client = genai.Client(api_key=self._api_key or "DUMMY_KEY")

    def _build_system_instruction(self) -> str:
        """Construct system instructions embedding the candidate profile.

        Returns:
            str: System instruction text.
        """
        allowed_str = (
            ", ".join(c.capitalize() for c in self.allowed_contracts)
            if self.allowed_contracts
            else "Alternance, Stage, CDI, CDD, Intérim"
        )

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
            "or in a department 13 city NOT listed in target hubs AND > 30 km "
            "from Allauch (13190) without full remote option -> is_match = "
            "False, score <= 30.\n"
            "   - If the role is predominantly Web Frontend / CMS / UI -> "
            "is_match = False, score <= 30.\n"
            "   - If the job requires a Senior, 'confirmé', or 'expérimenté' "
            "profile, or >= 3 years of experience (e.g., '3 à 4 ans', "
            "'3+ ans', '> 3 years') -> is_match = False, score <= 35.\n"
            f"   - If the job contract type is NOT in the allowed list "
            f"({allowed_str}) (e.g., Freelance, Indépendant, "
            "Portage salarial, Prestation externe) -> is_match = False, "
            "score <= 25.\n"
            "   - If the job description is a stub, placeholder, or too brief "
            "(< 300 characters of actual mission and stack) -> is_match = False, "
            "score <= 25, with reasoning stating 'Description trop sommaire'.\n"
            "   - If > 50% of the primary required technologies are missing -> "
            "is_match = False, score <= 40.\n"
            "2. Score calculation (0 to 100):\n"
            "   - 80-100: Exceptional match with AI agents, Python, C systems, "
            "DevOps, or SOC analyst roles in target locations/remote.\n"
            "   - 65-79: Good match with minor skill gaps easily bridgeable.\n"
            "   - 40-64: Partial match or secondary interest.\n"
            "   - 0-39: Definite mismatch or hit by dealbreakers.\n"
            "3. If the input company is generic, portal-based, or placeholder "
            "(e.g., 'HelloWork Listing', 'Entreprise', 'Unknown'), identify "
            "the real employer from the title or description and return it in "
            "the 'company' field.\n"
            "4. Extract the contract type into 'contract_type' using one of: "
            "'Alternance', 'Stage', 'CDI', 'CDD', 'Intérim', 'Freelance' "
            "(or null if absent/unclear).\n"
            "5. Return strict structured JSON adhering to the provided schema."
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

        breaker = get_circuit_breaker()
        raw_models = get_settings().get_gemini_model_cascade(
            self.model_name
        )
        models_to_try = breaker.filter_available_models(raw_models)

        if not models_to_try:
            raise RuntimeError(
                "All configured Gemini models are currently "
                "quarantined or exhausted."
            )

        last_err: Exception | None = None
        for current_model in models_to_try:
            for attempt in range(2):
                try:
                    # Appel asynchrone non-bloquant à l'API Gemini Flash
                    response = await self.client.aio.models.generate_content(
                        model=current_model,
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
                    last_err = err
                    quarantined = breaker.record_failure(current_model, err)
                    if quarantined:
                        # Modèle mort ou inexistant : passage immédiat au suivant
                        break

                    err_str = str(err)
                    if (
                        "503" in err_str
                        or "429" in err_str
                        or "UNAVAILABLE" in err_str
                    ) and attempt < 1:
                        await asyncio.sleep(1.5)
                        continue

                    # Passe au modèle alternatif en cas d'erreur de charge
                    print(
                        f"⚠️ [JobAnalyzer] Échec sur '{current_model}' ({err}), "
                        "bascule sur le modèle suivant..."
                    )
                    break

        raise RuntimeError(
            f"Failed to evaluate job posting after retries: {last_err}"
        )

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
        # 1. Règle éliminatoire stricte : description tronquée (< 300 caractères)
        clean_desc = description.strip()
        if len(clean_desc) < 300:
            stub_result = JobAnalysisResult(
                score=25,
                is_match=False,
                key_matches=[],
                missing_skills=["Description détaillée manquante"],
                reasoning=(
                    "Description trop sommaire / incomplète (< 300 caractères) : "
                    "impossible de vérifier les missions, le niveau d'expérience "
                    "et les prérequis techniques."
                ),
                company=company,
                contract_type=None,
            )
            return stub_result, False

        result = await self.analyze_job(
            title=title,
            company=company,
            description=description,
            platform=platform,
        )

        # 2. Règle éliminatoire stricte : type de contrat non autorisé (ex: Freelance)
        detected_ct = (result.contract_type or "").strip().lower()
        title_lower = title.lower()
        is_freelance = any(
            tok in detected_ct or tok in title_lower
            for tok in ("freelance", "independant", "indépendant", "portage")
        )
        has_invalid_contract = False
        if is_freelance:
            has_invalid_contract = True
        elif detected_ct and self.allowed_contracts:
            has_invalid_contract = not any(
                allowed in detected_ct for allowed in self.allowed_contracts
            )

        if has_invalid_contract:
            allowed_fmt = ", ".join(
                c.capitalize() for c in self.allowed_contracts
            )
            result.is_match = False
            result.score = min(result.score, 25)
            ct_label = result.contract_type or "Freelance"
            result.reasoning = (
                f"Type de contrat non souhaité ({ct_label}) : "
                f"seuls {allowed_fmt} sont autorisés."
            )
            return result, False

        # 3. Vérification du score et de l'adéquation globale
        is_qualified = result.is_match and result.score >= score_threshold

        # 4. Contrôle du ratio de compétences manquantes (ex: max 50%)
        total_skills = len(result.key_matches) + len(result.missing_skills)
        if total_skills > 0:
            missing_ratio = (len(result.missing_skills) / total_skills) * 100
            if missing_ratio > max_missing_skills_percent:
                is_qualified = False

        return result, is_qualified
