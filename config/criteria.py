"""Cross-platform parser for human-editable plain text search criteria."""

from pathlib import Path
import re
from typing import Any
from pydantic import BaseModel, Field

DEFAULT_CRITERIA_PATH: Path = (
    Path(__file__).resolve().parent.parent / "Profile" / "search_criteria.txt"
)


class SearchCriteria(BaseModel):
    """Structured representation of human-configured search preferences.

    Attributes:
        match_score_threshold: Minimal LLM score (0-100) to trigger Telegram alert.
        max_missing_skills: Maximum missing skills count before automatic rejection.
        max_missing_skills_percent: Max allowed ratio of missing skills.
        keywords: Whitelisted job position titles and skill keywords.
        locations: Target cities, departments, and remote policy tags.
        contracts: Allowed employment contract types.
        target_companies: Priority employers and organization names.
        custom_urls: Arbitrary external job board or ATS URLs to monitor.
    """

    match_score_threshold: int = Field(
        default=75,
        ge=0,
        le=100,
        description="Minimum Gemini Flash score (0-100) to alert candidate.",
    )
    max_missing_skills_percent: int = Field(
        default=50,
        ge=0,
        le=100,
        description="Maximum missing skill percentage allowed.",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="List of target search keywords.",
    )
    locations: list[str] = Field(
        default_factory=list,
        description="List of target geographical locations.",
    )
    contracts: list[str] = Field(
        default_factory=list,
        description="List of acceptable contract types.",
    )
    target_companies: list[str] = Field(
        default_factory=list,
        description="List of whitelisted target organizations.",
    )
    custom_urls: list[str] = Field(
        default_factory=list,
        description="List of custom or external job board URLs.",
    )


def parse_search_criteria_file(
    criteria_path: Path = DEFAULT_CRITERIA_PATH,
) -> SearchCriteria:
    """Parse a plain text search criteria file into a typed SearchCriteria model.

    Handles Windows (CRLF, UTF-8 BOM), macOS, and Linux line endings
    consistently across any standard text editor.

    Args:
        criteria_path: Path to the search_criteria.txt configuration file.

    Returns:
        SearchCriteria: Populated Pydantic criteria model.
    """
    if not criteria_path.exists():
        # Fallback sur les valeurs par défaut si le fichier n'existe pas
        return SearchCriteria()

    current_section = ""
    thresholds: dict[str, Any] = {}
    keywords: list[str] = []
    locations: list[str] = []
    contracts: list[str] = []
    target_companies: list[str] = []
    custom_urls: list[str] = []

    # Encodage utf-8-sig pour éliminer automatiquement le BOM du Bloc-notes Windows
    with open(criteria_path, "r", encoding="utf-8-sig") as f:
        for raw_line in f:
            # Suppression des espaces et retours chariot (\r\n et \n)
            line = raw_line.strip()

            # Ignore les lignes vides et commentaires (# ou ;)
            if not line or line.startswith(("#", ";")):
                continue

            # Détection d'en-tête de section, ex: [KEYWORDS]
            section_match = re.match(r"^\[([A-Za-z0-9_]+)\]$", line)
            if section_match:
                current_section = section_match.group(1).upper()
                continue

            # Traitement selon la section active
            if current_section == "THRESHOLDS":
                if "=" in line:
                    key, val = line.split("=", 1)
                    k = key.strip().lower()
                    v = val.split("#")[0].split(";")[0].strip()
                    try:
                        thresholds[k] = int(v)
                    except ValueError:
                        continue

            elif current_section == "KEYWORDS":
                item = line.split("#")[0].split(";")[0].strip()
                if item and item not in keywords:
                    keywords.append(item)

            elif current_section == "LOCATIONS":
                item = line.split("#")[0].split(";")[0].strip()
                if item and item not in locations:
                    locations.append(item)

            elif current_section == "CONTRACTS":
                item = line.split("#")[0].split(";")[0].strip()
                if item and item not in contracts:
                    contracts.append(item)

            elif current_section == "TARGET_COMPANIES":
                item = line.split("#")[0].split(";")[0].strip()
                if item and item not in target_companies:
                    target_companies.append(item)

            elif current_section == "CUSTOM_URLS":
                item = line.split("#")[0].split(";")[0].strip()
                if item and item.startswith("http") and item not in custom_urls:
                    custom_urls.append(item)

    return SearchCriteria(
        match_score_threshold=thresholds.get("match_score_threshold", 75),
        max_missing_skills_percent=thresholds.get(
            "max_missing_skills_percent", 50
        ),
        keywords=keywords,
        locations=locations,
        contracts=contracts,
        target_companies=target_companies,
        custom_urls=custom_urls,
    )
