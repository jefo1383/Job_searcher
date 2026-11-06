"""Automated candidate profile generator from PDF and Word resumes using Gemini."""

from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

from google import genai
from google.genai import types

from config.criteria import SearchCriteria, parse_search_criteria_file
from config.settings import Settings, get_settings

# Namespaces XML standard OpenXML pour les fichiers Word .docx
DOCX_NAMESPACE: dict[str, str] = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
}


def find_latest_resume(profile_dir: Path) -> Path | None:
    """Find the most recently updated CV file (.pdf or .docx) in the folder.

    Args:
        profile_dir: Path to directory containing candidate documents.

    Returns:
        Path | None: Path to newest resume file or None if none found.
    """
    if not profile_dir.exists():
        return None

    candidates: list[Path] = []
    for ext in ("*.pdf", "*.docx", "*.doc"):
        for file_path in profile_dir.glob(ext):
            # Ignore les fichiers temporaires Word débutant par ~$ ou .
            if file_path.name.startswith(("~$", ".")):
                continue
            candidates.append(file_path)

    if not candidates:
        return None

    # Sélection du fichier le plus récent selon la date de modification
    return max(candidates, key=lambda p: p.stat().st_mtime)


def extract_text_from_docx(docx_path: Path) -> str:
    """Extract plain text paragraphs from a Word .docx document using stdlib.

    Args:
        docx_path: Path to .docx file.

    Returns:
        str: Cleaned concatenated text of all document paragraphs.
    """
    paragraphs: list[str] = []
    try:
        with zipfile.ZipFile(docx_path, "r") as z:
            if "word/document.xml" not in z.namelist():
                return ""
            xml_content = z.read("word/document.xml")
            root = ET.fromstring(xml_content)

            for p in root.iterfind(".//w:p", DOCX_NAMESPACE):
                texts = [
                    node.text
                    for node in p.iterfind(".//w:t", DOCX_NAMESPACE)
                    if node.text
                ]
                if texts:
                    paragraphs.append("".join(texts))

    except Exception as err:
        print(f"⚠️ [ProfileBuilder] Error parsing docx file {docx_path}: {err}")
        return ""

    return "\n".join(paragraphs).strip()


class ProfileBuilder:
    """Builds and refreshes config/profile.md from a dropped PDF or Word resume.

    Attributes:
        settings: Application settings.
        client: Google GenAI client instance.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize ProfileBuilder.

        Args:
            settings: Optional custom Settings instance.
            api_key: Optional Gemini API key.
        """
        self.settings = settings or get_settings()
        resolved_key = (
            api_key
            or self.settings.gemini_api_key.get_secret_value()
            or "DUMMY_KEY"
        )
        self.client = genai.Client(api_key=resolved_key)

    def is_update_needed(self) -> tuple[bool, Path | None]:
        """Check if profile.md needs regeneration from the newest resume.

        Returns:
            tuple[bool, Path | None]: (needs_update, latest_resume_path).
        """
        resume_file = find_latest_resume(self.settings.profile_dir)
        profile_md = self.settings.candidate_profile_path

        # Si aucun CV n'est présent dans le dossier
        if resume_file is None:
            return False, None

        # Si profile.md n'existe pas encore
        if not profile_md.exists():
            return True, resume_file

        # Si le CV a été modifié plus récemment que profile.md
        if resume_file.stat().st_mtime > profile_md.stat().st_mtime:
            return True, resume_file

        return False, resume_file

    async def build_profile_markdown(
        self,
        resume_path: Path,
        criteria: SearchCriteria | None = None,
    ) -> str:
        """Analyze resume and synthesize complete profile.md using Gemini Flash.

        Args:
            resume_path: Path to source .pdf or .docx CV file.
            criteria: Optional search criteria to integrate into directives.

        Returns:
            str: Generated structured Markdown profile.
        """
        active_criteria = criteria or parse_search_criteria_file(
            self.settings.search_criteria_path
        )

        # Préparation du contenu du prompt selon le format du CV
        contents: list[Any] = []

        if resume_path.suffix.lower() == ".pdf":
            # Transmission directe du PDF en entrée multimodale
            pdf_bytes = resume_path.read_bytes()
            pdf_part = types.Part.from_bytes(
                data=pdf_bytes,
                mime_type="application/pdf",
            )
            contents.append(pdf_part)
            resume_context_note = (
                "The candidate's full resume is provided as an attached PDF."
            )
        else:
            # Extraction textuelle du document Word
            extracted_text = extract_text_from_docx(resume_path)
            resume_context_note = (
                f"Candidate resume content (extracted from Word document):\n\n"
                f"{extracted_text}"
            )

        target_roles = (
            ", ".join(active_criteria.keywords)
            or "Software Engineer, DevOps, AI, Cyber"
        )
        target_locs = (
            ", ".join(active_criteria.locations)
            or "13, Marseille, Aix-en-Provence, Remote"
        )
        allowed_contracts = (
            ", ".join(active_criteria.contracts)
            or "Alternance, CDI, CDD, Intérim"
        )
        target_comps = (
            ", ".join(active_criteria.target_companies)
            or "Airbus, Thales, CMA CGM, FDJ, BPCE, Vinci, Orange"
        )

        instructions = (
            "You are an expert executive talent assessor and technical "
            "profile architect. Your mission is to analyze the provided "
            "candidate resume and synthesize a thorough, highly structured "
            "Markdown profile document ('profile.md') to be used by automated "
            "LLM evaluators and application generators.\n\n"
            f"{resume_context_note}\n\n"
            "CRITERIA & TARGETING DIRECTIVES TO INTEGRATE:\n"
            f"- Target Roles: {target_roles}\n"
            f"- Target Locations: {target_locs}\n"
            f"- Allowed Contracts: {allowed_contracts}\n"
            f"- Target Companies: {target_comps}\n"
            f"- Max Missing Skills Percent: "
            f"{active_criteria.max_missing_skills_percent}%\n\n"
            "REQUIRED OUTPUT MARKDOWN STRUCTURE:\n"
            "# Candidate Profile & Job Hunter Directives\n"
            "## 1. Candidate Identity & Professional Overview\n"
            "## 2. Core Technical Competencies\n"
            "## 3. Major Engineering Projects & Proof of Work\n"
            "## 4. Prior Professional Background & Transferable Skills\n"
            "## 5. Job Search Targeting Preferences & Constraints\n\n"
            "Output ONLY clean Markdown formatted strictly adhering to this "
            "structure, with no conversational fluff or markdown code fence."
        )
        contents.append(instructions)

        response = await self.client.aio.models.generate_content(
            model=self.settings.gemini_model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.2,
            ),
        )

        markdown_text = response.text or ""
        # Nettoyage d'éventuels backticks de bloc ```markdown
        cleaned = re.sub(
            r"^```markdown\s*", "", markdown_text, flags=re.IGNORECASE
        )
        cleaned = re.sub(r"```$", "", cleaned.strip())
        return cleaned.strip()

    async def sync_profile(self, force: bool = False) -> bool:
        """Check and update profile.md from the newest resume if needed.

        Args:
            force: If True, force regeneration regardless of file timestamps.

        Returns:
            bool: True if profile was regenerated and written, False otherwise.
        """
        needs_update, resume_file = self.is_update_needed()
        if not force and not needs_update:
            return False

        if resume_file is None:
            print(
                "ℹ️ [ProfileBuilder] Aucun fichier de CV (.pdf ou .docx) "
                "trouvé dans Profile/."
            )
            return False

        print(
            f"📄 [ProfileBuilder] Analyse du CV '{resume_file.name}' "
            f"et régénération de profile.md via Gemini Flash..."
        )
        profile_content = await self.build_profile_markdown(resume_file)

        # Écriture du fichier profile.md
        target_path = self.settings.candidate_profile_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(profile_content, encoding="utf-8")

        print(f"✓ [ProfileBuilder] '{target_path.name}' généré avec succès !")
        return True
