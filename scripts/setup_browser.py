"""Script d'initialisation et de configuration hybride de Chromium pour Playwright.

Détecte si un binaire Chromium système est disponible pour lier ses exécutables
dans le cache Playwright, évitant ainsi le téléchargement lourd et les erreurs
de routage IPv6 (spécifiques aux clusters 42). En l'absence de binaire système,
déclenche l'installation officielle en forçant la résolution DNS IPv4.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Final

# Binaires candidats prioritaires sur les systèmes Linux / Unix
CANDIDATE_BROWSERS: Final[tuple[str, ...]] = (
    "/usr/bin/chromium-browser",
    "/usr/bin/chromium",
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
)


def _find_system_browser() -> Path | None:
    """Recherche un binaire Chromium ou Chrome valide sur le système.

    Returns:
        Path | None: Le chemin absolu vers le binaire ou None si absent.
    """
    for candidate in CANDIDATE_BROWSERS:
        candidate_path = Path(candidate)
        if candidate_path.is_file() and os.access(candidate_path, os.X_OK):
            return candidate_path

    # Recherche générique via le PATH système
    for binary_name in ("chromium-browser", "chromium", "google-chrome"):
        which_path = shutil.which(binary_name)
        if which_path:
            return Path(which_path)

    return None


def _get_playwright_revisions() -> tuple[str, str]:
    """Récupère les révisions attendues de chromium depuis browsers.json.

    Returns:
        tuple[str, str]: (révision chromium, révision chromium-headless-shell).
    """
    default_revision = "1248"
    try:
        import playwright

        playwright_dir = Path(playwright.__file__).parent
        browsers_json = (
            playwright_dir / "driver" / "package" / "browsers.json"
        )
        if browsers_json.is_file():
            data = json.loads(browsers_json.read_text(encoding="utf-8"))
            browsers = data.get("browsers", [])
            cr_rev = next(
                (
                    str(b["revision"])
                    for b in browsers
                    if b.get("name") == "chromium"
                ),
                default_revision,
            )
            hs_rev = next(
                (
                    str(b["revision"])
                    for b in browsers
                    if b.get("name") == "chromium-headless-shell"
                ),
                cr_rev,
            )
            return cr_rev, hs_rev
    except Exception:
        pass
    return default_revision, default_revision


def _link_system_browser(system_browser: Path) -> None:
    """Crée les liens symboliques dans le cache Playwright vers le binaire système.

    Args:
        system_browser: Chemin vers le binaire Chromium ou Chrome.
    """
    cache_base = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if cache_base:
        pw_cache = Path(cache_base)
    else:
        pw_cache = Path.home() / ".cache" / "ms-playwright"

    cr_rev, hs_rev = _get_playwright_revisions()

    links_to_create = (
        (pw_cache / f"chromium-{cr_rev}" / "chrome-linux64", "chrome"),
        (
            pw_cache
            / f"chromium_headless_shell-{hs_rev}"
            / "chrome-headless-shell-linux64",
            "chrome-headless-shell",
        ),
    )

    for target_dir, link_name in links_to_create:
        target_dir.mkdir(parents=True, exist_ok=True)
        link_path = target_dir / link_name
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
        link_path.symlink_to(system_browser)

    print(
        f"✅ Navigateur système lié : {system_browser} -> {pw_cache} "
        f"(révisions {cr_rev}/{hs_rev}, 0 Mo téléchargé)."
    )


def _install_playwright_browser() -> None:
    """Télécharge Chromium via Playwright en forçant l'IPv4 contre ENETUNREACH."""
    print(
        "🌐 Aucun Chromium système détecté. "
        "Téléchargement via Playwright (IPv4 forcé)..."
    )
    env = os.environ.copy()
    env["NODE_OPTIONS"] = "--dns-result-order=ipv4first"

    cmd = ["uv", "run", "playwright", "install", "chromium"]
    subprocess.check_call(cmd, env=env)
    print("✅ Installation de Chromium par Playwright terminée.")


def main() -> None:
    """Point d'entrée du script d'assurance du navigateur Playwright."""
    system_browser = _find_system_browser()
    if system_browser:
        _link_system_browser(system_browser)
    else:
        _install_playwright_browser()


if __name__ == "__main__":
    main()
