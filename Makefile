# ==============================================================================
# Makefile - Job Hunter Bot
# ==============================================================================

# Variables réglables (surchargeables en ligne de commande : make once ARGS="--limit 10")
UV ?= uv
PYTHON := $(UV) run python
ENTRY := main.py
APP_NAME := job-hunter
BUILD_DIR := build
DIST_DIR := dist
INSTALL_DIR ?= $(HOME)/.local/bin
VENV_DIR := .venv
ARGS ?=
PYINSTALLER_FLAGS ?= --onefile --name $(APP_NAME)

.DEFAULT_GOAL := help

.PHONY: help setup install run once bot indeed lint clean fclean re

help: ## Affiche l'aide des commandes disponibles
	@echo "Commandes disponibles :"
	@echo "  make setup    - Installe l'environnement uv et Chromium (Playwright)"
	@echo "  make install  - Compile et installe le futur binaire PyInstaller"
	@echo "  make run      - Lance la boucle complète (collecte périodique + bot)"
	@echo "  make once     - Exécute un cycle unique (--once)"
	@echo "  make bot      - Lance uniquement le bot Telegram (--bot-only)"
	@echo "  make indeed   - Lance le mode navigateur interactif Indeed (--indeed)"
	@echo "  make lint     - Vérifie le code avec flake8 et mypy"
	@echo "  make clean    - Supprime les caches Python, mypy et artefacts build/"
	@echo "  make fclean   - Exécute clean puis supprime dist/ et .venv (conserve SQLite et .env)"
	@echo "  make re       - Réinitialise complètement l'environnement (fclean + setup)"

setup: ## Prépare l'environnement virtuel uv et le navigateur Chromium
	$(UV) sync
	$(PYTHON) scripts/setup_browser.py

install: ## Compile et installe le package PyInstaller dans INSTALL_DIR
	@if ! $(PYTHON) -c "import PyInstaller" >/dev/null 2>&1; then \
		echo "ℹ️ PyInstaller n'est pas encore installé dans $(VENV_DIR)."; \
		echo "   Lors de l'étape packaging, exécutez : $(UV) add --dev pyinstaller"; \
		exit 1; \
	fi
	$(UV) run pyinstaller $(PYINSTALLER_FLAGS) --workpath $(BUILD_DIR) --distpath $(DIST_DIR) $(ENTRY)
	mkdir -p $(INSTALL_DIR)
	install -m 0755 $(DIST_DIR)/$(APP_NAME) $(INSTALL_DIR)/$(APP_NAME)
	@echo "✅ Binaire installé : $(INSTALL_DIR)/$(APP_NAME)"

run: ## Lance la boucle complète (veille périodique + écoute Telegram)
	$(PYTHON) $(ENTRY) $(ARGS)

once: ## Exécute un seul cycle de collecte et d'évaluation (--once)
	$(PYTHON) $(ENTRY) --once $(ARGS)

bot: ## Lance uniquement le bot Telegram interactif (--bot-only)
	$(PYTHON) $(ENTRY) --bot-only $(ARGS)

indeed: ## Ouvre Chrome manuellement pour scraper Indeed (--indeed)
	$(PYTHON) $(ENTRY) --indeed $(ARGS)

lint: ## Exécute flake8 et mypy sur l'ensemble du projet
	$(UV) run flake8 .
	$(UV) run mypy .

clean: ## Nettoie les caches et fichiers intermédiaires de build
	find . -path ./$(VENV_DIR) -prune -o -type d -name "__pycache__" -exec rm -rf {} +
	find . -path ./$(VENV_DIR) -prune -o -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
	rm -rf .mypy_cache .pytest_cache $(BUILD_DIR) *.spec

fclean: clean ## Nettoie build, caches, dist/ et .venv (préserve data/ et .env)
	rm -rf $(DIST_DIR) $(VENV_DIR)
	@echo "✅ fclean terminé (base SQLite et .env conservés)."

re: fclean setup ## Reconstruit entièrement l'environnement de travail
