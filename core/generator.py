"""Application document and pitch generator module using Google GenAI SDK."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any
import unicodedata

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from config.settings import get_settings


def sanitize_filename(name: str) -> str:
    """Sanitize string for safe directory and file names.

    Args:
        name: Raw identifier string.

    Returns:
        str: Lowercase alphanumeric string with underscores.
    """
    decomposed = unicodedata.normalize("NFKD", name)
    cleaned = "".join(c for c in decomposed if not unicodedata.combining(c))
    cleaned = re.sub(r"[^\w\s-]", "", cleaned).strip().lower()
    return re.sub(r"[\s-]+", "_", cleaned)


def export_application_markdown(
    job_id: int,
    company: str,
    title: str,
    url: str,
    score: int | None,
    contract_type: str | None,
    application: "TailoredApplication",
    key_matches: list[str] | None = None,
    missing_skills: list[str] | None = None,
    reasoning: str | None = None,
    base_dir: Path | str = "Applications",
    overwrite: bool = False,
) -> Path:
    """Export tailored application document, pitch, and metadata to Markdown.

    Args:
        job_id: SQLite primary key of the offer.
        company: Employing organization name.
        title: Position or role title.
        url: Original job offer URL.
        score: Relevance match score (0-100).
        contract_type: Extracted contract category.
        application: Generated application package with pitch and cover letter.
        key_matches: List of confirmed skill matches.
        missing_skills: List of potential skill compromises.
        reasoning: Gemini evaluation rationale.
        base_dir: Root directory for exported applications (default: 'Applications').
        overwrite: Whether to overwrite an existing lettre_motivation.md file.

    Returns:
        Path: Filesystem path to the written lettre_motivation.md file.
    """
    root = Path(base_dir)
    safe_comp = sanitize_filename(company) or "entreprise"
    folder_name = f"{safe_comp}_{job_id}"
    app_folder = root / folder_name
    app_folder.mkdir(parents=True, exist_ok=True)

    file_path = app_folder / "lettre_motivation.md"
    if file_path.exists() and not overwrite:
        return file_path

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    score_str = f"{score}/100" if score is not None else "Non noté"
    contract_str = contract_type or "Non précisé"

    matches_md = (
        "\n".join(f"- {m}" for m in key_matches)
        if key_matches
        else "- Non spécifié"
    )
    missing_md = (
        "\n".join(f"- {m}" for m in missing_skills)
        if missing_skills
        else "- Aucune lacune majeure détectée"
    )
    reasoning_md = reasoning or "Non spécifiée."

    content = (
        f"# Candidature — {company} — {title}\n\n"
        f"- **Identifiant de l'offre :** #{job_id}\n"
        f"- **Score Gemini :** {score_str}\n"
        f"- **Type de contrat :** {contract_str}\n"
        f"- **URL de l'offre :** {url}\n"
        f"- **Date de génération :** {now_iso}\n"
        f"- **Langue détectée :** {application.language.upper()}\n"
        f"- **Objet suggéré :** {application.subject_line}\n\n"
        f"---\n\n"
        f"## 📌 Pitch d'accroche (Accroche rapide)\n\n"
        f"{application.quick_pitch}\n\n"
        f"---\n\n"
        f"## 📄 Lettre de motivation personnalisée\n\n"
        f"{application.full_cover_letter}\n\n"
        f"---\n\n"
        f"## 🔍 Synthèse d'évaluation\n\n"
        f"*{reasoning_md}*\n\n"
        f"### Points forts retenus\n"
        f"{matches_md}\n\n"
        f"### Compétences à développer / compromis\n"
        f"{missing_md}\n"
    )

    file_path.write_text(content, encoding="utf-8")
    return file_path


class TailoredApplication(BaseModel):
    """Structured application documents produced for a qualified job offer.

    Attributes:
        language: Detected output language code ('fr' or 'en').
        subject_line: Catchy and professional application email/message subject.
        quick_pitch: Concise 3-4 sentence hook for Telegram alerts or LinkedIn.
        full_cover_letter: Comprehensive tailored cover letter ready for ATS.
    """

    language: str = Field(
        ...,
        description="Target output language code ('fr' or 'en').",
    )
    subject_line: str = Field(
        ...,
        description="Targeted application subject line.",
    )
    quick_pitch: str = Field(
        ...,
        description="Concise 3-4 sentence hook for Telegram or quick messaging.",
    )
    full_cover_letter: str = Field(
        ...,
        description="Structured cover letter tailored to the specific position.",
    )


class JobGenerator:
    """Asynchronous generator crafting contextual pitches and cover letters.

    Attributes:
        model_name: Gemini model identifier.
        profile_content: Candidate profile loaded from Markdown.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        profile_path: Path | None = None,
    ) -> None:
        """Initialize the generator with settings and candidate profile.

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
        """Construct system instructions enforcing natural tone and B2 English.

        Returns:
            str: System instruction text.
        """
        return (
            "You are an expert technical career writer and software engineer "
            "mentor crafting high-impact, human, and authentic application "
            "documents.\n\n"
            f"=== CANDIDATE PROFILE ===\n"
            f"{self.profile_content}\n"
            f"=========================\n\n"
            "STYLE & TONE INSTRUCTIONS (MANDATORY):\n"
            "1. NO AI CLICHÉS OR CORNY PHRASES:\n"
            "   - Strictly ban: 'I am thrilled/excited to apply', 'delve', "
            "'testament to', 'tapestry', 'myriad', 'pleased to submit', "
            "'esteemed company', 'passionate developer', 'fast-paced world'.\n"
            "   - Start directly with tangible relevance, genuine technical "
            "curiosity, or shared engineering challenges.\n"
            "2. LANGUAGE ADAPTATION & ENGLISH B2 LEVEL:\n"
            "   - Detect the language of the job posting (French or English).\n"
            "   - If the job is in French, write in natural, professional French "
            "without pompous administrative formulas.\n"
            "   - If the job is in English, write in an authentic, natural B2 "
            "level (professional working proficiency). Use clear, direct, "
            "everyday software engineering vocabulary. Avoid overly literary "
            "or ornate vocabulary.\n"
            "3. CONTENT STRUCTURE:\n"
            "   - quick_pitch: 3-4 punchy sentences summarizing why the "
            "candidate's background (School 42 + 10y project management + "
            "concrete projects) creates immediate value for this specific role.\n"
            "   - full_cover_letter: 3-4 structured paragraphs:\n"
            "       * Hook: Immediate alignment with the company's stack/mission.\n"
            "       * Proof of Work: Highlight 1-2 concrete achievements "
            "(Cyber AI Hackathon with CMA CGM/Mistral, autonomous agents, "
            "concurrency in C, or prior management).\n"
            "       * Missing skills handling: If minor gaps exist, pivot "
            "constructively to first-principles learning speed demonstrated "
            "at School 42.\n"
            "       * Call to Action: Professional and proactive closing.\n"
            "4. Return strictly valid structured JSON matching the schema."
        )

    async def generate_application(
        self,
        title: str,
        company: str,
        description: str,
        key_matches: list[str] | None = None,
        missing_skills: list[str] | None = None,
        platform: str = "",
    ) -> TailoredApplication:
        """Generate tailored application documents for a matched job posting.

        Args:
            title: Job position title.
            company: Hiring organization name.
            description: Job description text.
            key_matches: Validated matching skills from analyzer.
            missing_skills: Identified competency gaps from analyzer.
            platform: Job portal name.

        Returns:
            TailoredApplication: Populated application documents.

        Raises:
            ValueError: If the Google AI Studio API key is not configured.
            RuntimeError: If LLM generation or parsing fails.
        """
        if not self._api_key or self._api_key == "DUMMY_KEY":
            raise ValueError(
                "GEMINI_API_KEY is not set. Please provide a valid key in "
                ".env or pass it to JobGenerator."
            )

        matches_summary = (
            ", ".join(key_matches) if key_matches else "Not specified"
        )
        gaps_summary = (
            ", ".join(missing_skills) if missing_skills else "None identified"
        )

        user_prompt = (
            f"TARGET JOB SPECIFICATIONS:\n"
            f"- Company: {company}\n"
            f"- Role: {title}\n"
            f"- Platform: {platform or 'Direct portal'}\n"
            f"- Key Confirmed Matches: {matches_summary}\n"
            f"- Identified Compromises / Missing Skills: {gaps_summary}\n"
            f"- Job Description:\n{description}\n"
        )

        # Température calibrée à 0.4 pour un style naturel et fluide
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TailoredApplication,
            temperature=0.4,
            system_instruction=self._build_system_instruction(),
        )

        models_to_try = [self.model_name]
        for fallback in ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]:
            if fallback not in models_to_try:
                models_to_try.append(fallback)

        last_err: Exception | None = None
        for current_model in models_to_try:
            for attempt in range(2):
                try:
                    # Appel asynchrone non-bloquant
                    response = await self.client.aio.models.generate_content(
                        model=current_model,
                        contents=user_prompt,
                        config=config,
                    )

                    if hasattr(response, "parsed") and response.parsed is not None:
                        parsed_result: Any = response.parsed
                        if isinstance(parsed_result, TailoredApplication):
                            return parsed_result
                        if isinstance(parsed_result, dict):
                            return TailoredApplication.model_validate(
                                parsed_result
                            )

                    if response.text:
                        return TailoredApplication.model_validate_json(
                            response.text
                        )

                    raise RuntimeError("Gemini returned an empty response body.")

                except Exception as err:
                    last_err = err
                    err_str = str(err)
                    if (
                        "503" in err_str
                        or "429" in err_str
                        or "UNAVAILABLE" in err_str
                    ) and attempt < 1:
                        await asyncio.sleep(1.5)
                        continue

                    # Passe au modèle de secours
                    break

        raise RuntimeError(
            f"Failed to generate tailored documents after retries: {last_err}"
        )
