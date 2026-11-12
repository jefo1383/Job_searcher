"""Hybrid rule-based and LLM screening question solver for ATS forms."""

import asyncio
from pathlib import Path
import re
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from config.settings import get_settings


class QuestionAnswerResult(BaseModel):
    """Structured answer returned for an application screening question.

    Attributes:
        answer: Exact value to input or exact option label to select.
        confidence: Confidence score between 0.0 and 1.0.
        reasoning: Brief explanation justifying the chosen response.
    """

    answer: str = Field(
        ...,
        description="Exact value to fill or exact option text to select.",
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence level for the proposed answer (0.0-1.0).",
    )
    reasoning: str = Field(
        default="",
        description="Short rationale based on candidate profile.",
    )


class ScreeningQuestionSolver:
    """Solves ATS and Easy Apply screening questions using rules and Gemini.

    Combines deterministic local keyword rules for common administrative
    questions with Gemini Flash contextual inference against profile.md.

    Attributes:
        model_name: Primary Gemini model identifier.
        min_confidence: Minimum LLM confidence threshold to accept an answer.
        profile_content: Loaded candidate profile text.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        profile_path: Path | None = None,
        min_confidence: float = 0.5,
    ) -> None:
        """Initialize the screening question solver.

        Args:
            api_key: Optional Google AI Studio API key.
            model_name: Optional Gemini model identifier.
            profile_path: Optional path to candidate profile Markdown.
            min_confidence: Threshold below which LLM answers are discarded.
        """
        settings = get_settings()
        resolved_key = (
            api_key
            if api_key is not None
            else settings.gemini_api_key.get_secret_value()
        )
        self._api_key = resolved_key
        self.model_name = model_name or settings.gemini_model
        self.min_confidence = min_confidence

        target_profile = profile_path or settings.candidate_profile_path
        if target_profile.exists():
            self.profile_content = target_profile.read_text(encoding="utf-8")
        else:
            self.profile_content = ""

        # Client GenAI initialisé uniquement si une clé valide est présente
        self.client: genai.Client | None = (
            genai.Client(api_key=self._api_key)
            if self._api_key and self._api_key != "DUMMY_KEY"
            else None
        )

    @staticmethod
    def _pick_matching_option(
        desired_tokens: list[str],
        options: list[str] | None,
    ) -> str | None:
        """Select the best matching option label from available choices.

        Args:
            desired_tokens: Ordered lowercase keyword candidates to match.
            options: Available choice labels from a select or radio group.

        Returns:
            str | None: Exact original option string if matched, else None.
        """
        if not options:
            return None

        for token in desired_tokens:
            for opt in options:
                if opt.strip().lower() == token:
                    return opt
            for opt in options:
                if token in opt.strip().lower():
                    return opt
        return None

    def _match_local_rules(
        self,
        question: str,
        field_type: str = "text",
        options: list[str] | None = None,
    ) -> QuestionAnswerResult | None:
        """Evaluate fast deterministic rules before calling Gemini Flash.

        Args:
            question: Screening question text or label.
            field_type: Input kind ('text', 'number', 'select', 'radio').
            options: Optional list of selectable labels.

        Returns:
            QuestionAnswerResult | None: Immediate answer if rule matched.
        """
        q_lower = question.lower()

        # 1. Besoin de visa / parrainage (sponsorship)
        if any(
            kw in q_lower
            for kw in ("sponsor", "parrainage", "visa", "titre de séjour")
        ):
            matched = self._pick_matching_option(
                ["non", "no", "aucun"], options
            )
            default_no = "Non" if "fr" in q_lower or "ê" in q_lower else "No"
            return QuestionAnswerResult(
                answer=matched or default_no,
                confidence=0.98,
                reasoning="Citoyen français, aucun besoin de parrainage visa.",
            )

        # 2. Autorisation légale de travailler en France / UE
        if any(
            kw in q_lower
            for kw in (
                "autorisé à travailler",
                "authorized to work",
                "legally authorized",
                "work authorization",
                "droit au travail",
            )
        ):
            matched = self._pick_matching_option(["oui", "yes"], options)
            return QuestionAnswerResult(
                answer=matched or ("Oui" if "autorisé" in q_lower else "Yes"),
                confidence=0.98,
                reasoning="Autorisation de travail permanente en France/UE.",
            )

        # 3. Permis B / Véhicule
        if any(
            kw in q_lower
            for kw in ("permis b", "permis de conduire", "driver", "driving")
        ):
            matched = self._pick_matching_option(["oui", "yes"], options)
            return QuestionAnswerResult(
                answer=matched or ("Oui" if "permis" in q_lower else "Yes"),
                confidence=0.95,
                reasoning="Règle locale permis B.",
            )

        # 4. Disponibilité immédiate / préavis
        if any(
            kw in q_lower
            for kw in (
                "préavis",
                "notice period",
                "disponibilité",
                "availability",
                "immédiate",
            )
        ):
            if field_type == "number":
                return QuestionAnswerResult(
                    answer="0",
                    confidence=0.95,
                    reasoning="Disponibilité immédiate (0 jour/mois de préavis).",
                )
            matched = self._pick_matching_option(
                ["immédiat", "immediate", "oui", "yes", "0"], options
            )
            if matched or not options:
                return QuestionAnswerResult(
                    answer=matched or "Immédiate",
                    confidence=0.9,
                    reasoning="Disponibilité immédiate configurée.",
                )

        # 5. Téléphone / email parfois demandé en question libre
        if "linkedin" in q_lower and field_type == "text":
            return QuestionAnswerResult(
                answer="https://www.linkedin.com/in/jérémy-foeller-95207a70",
                confidence=0.99,
                reasoning="URL LinkedIn du profil.",
            )

        if "github" in q_lower and field_type == "text":
            return QuestionAnswerResult(
                answer="https://github.com/jefo1383",
                confidence=0.99,
                reasoning="URL GitHub du profil.",
            )

        return None

    async def solve_question(
        self,
        question: str,
        field_type: str = "text",
        options: list[str] | None = None,
        job_context: str = "",
    ) -> str | None:
        """Determine the best answer for a screening question.

        Checks local deterministic rules first, then falls back to Gemini
        Flash using the candidate profile.

        Args:
            question: Visible question label extracted from the form.
            field_type: Input category ('text', 'number', 'select', 'radio').
            options: Available choices when field_type is 'select' or 'radio'.
            job_context: Optional title/company string for better accuracy.

        Returns:
            str | None: Resolved answer string, or None if unsolved/low confidence.
        """
        if not question.strip():
            return None

        # Phase 1 : Règles déterministes locales ultra-rapides
        rule_result = self._match_local_rules(
            question=question,
            field_type=field_type,
            options=options,
        )
        if rule_result is not None:
            return rule_result.answer

        # Phase 2 : Inférence contextuelle via Gemini Flash
        if self.client is None:
            return None

        llm_result = await self._solve_with_llm(
            question=question,
            field_type=field_type,
            options=options,
            job_context=job_context,
        )
        if llm_result is None or llm_result.confidence < self.min_confidence:
            return None

        answer = llm_result.answer.strip()
        if not answer:
            return None

        # Normalisation stricte pour les champs numériques
        if field_type == "number":
            digits = re.search(r"\d+", answer)
            return digits.group(0) if digits else None

        # Vérification stricte de correspondance si une liste d'options existe
        if options:
            exact = self._pick_matching_option([answer.lower()], options)
            return exact

        return answer

    async def _solve_with_llm(
        self,
        question: str,
        field_type: str,
        options: list[str] | None,
        job_context: str,
    ) -> QuestionAnswerResult | None:
        """Query Gemini Flash with structured output to answer an unknown prompt.

        Args:
            question: Form label text.
            field_type: Expected HTML input category.
            options: Selectable choice strings if applicable.
            job_context: Optional company and job title context.

        Returns:
            QuestionAnswerResult | None: Parsed structured response or None.
        """
        if self.client is None:
            return None

        options_block = (
            "AVAILABLE OPTIONS (you MUST return one of these exact strings):\n"
            + "\n".join(f"- {opt}" for opt in options)
            if options
            else "NO PREDEFINED OPTIONS (free-text or numeric input)."
        )

        format_rule = (
            "Return ONLY an integer string (e.g. '1', '2', '0')."
            if field_type == "number"
            else "Return a concise, accurate, honest value matching the question."
        )

        system_instruction = (
            "You are assisting a candidate filling an online job application "
            "form. Answer screening questions honestly and strategically "
            "using ONLY facts from the candidate profile below.\n\n"
            "RULES:\n"
            "1. Never invent degrees or certifications the candidate lacks.\n"
            "2. For software/project management experience, reflect both 42 "
            "engineering projects and 10 years of prior management/ops "
            "where applicable, staying realistic for junior tech stacks.\n"
            f"3. {format_rule}\n"
            "4. If the question cannot be answered safely (e.g. sensitive "
            "personal ID), set confidence <= 0.2.\n\n"
            f"=== CANDIDATE PROFILE ===\n{self.profile_content}\n"
        )

        user_prompt = (
            f"JOB CONTEXT: {job_context or 'Standard application'}\n"
            f"FIELD TYPE: {field_type}\n"
            f"QUESTION: {question}\n"
            f"{options_block}\n"
        )

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=QuestionAnswerResult,
            temperature=0.1,
            system_instruction=system_instruction,
        )

        models_to_try = [self.model_name]
        for fallback in ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]:
            if fallback not in models_to_try:
                models_to_try.append(fallback)

        for current_model in models_to_try:
            for attempt in range(2):
                try:
                    response = await self.client.aio.models.generate_content(
                        model=current_model,
                        contents=user_prompt,
                        config=config,
                    )
                    if hasattr(response, "parsed") and response.parsed:
                        parsed: Any = response.parsed
                        if isinstance(parsed, QuestionAnswerResult):
                            return parsed
                        if isinstance(parsed, dict):
                            return QuestionAnswerResult.model_validate(parsed)

                    if response.text:
                        return QuestionAnswerResult.model_validate_json(
                            response.text
                        )
                    return None
                except Exception as err:
                    err_str = str(err)
                    if (
                        "503" in err_str
                        or "429" in err_str
                        or "UNAVAILABLE" in err_str
                    ) and attempt < 1:
                        await asyncio.sleep(1.0)
                        continue
                    break

        return None
