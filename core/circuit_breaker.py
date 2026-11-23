"""Circuit breaker pattern for resilient Gemini LLM model cascades."""

from typing import Final

# Motifs identifiant un dépassement de quota journalier irrécupérable à court terme
DAILY_QUOTA_PATTERNS: Final[tuple[str, ...]] = (
    "limit: 20",
    "requests per day",
    "per day",
    "generaterequestsperday",
    "free_tier_requests",
)

# Motifs identifiant un modèle inexistant ou déprécié dans le catalogue API
NOT_FOUND_PATTERNS: Final[tuple[str, ...]] = (
    "404",
    "not found",
    "not_found",
)


class ModelCircuitBreaker:
    """Tracks and blacklists exhausted or non-existent Gemini models."""

    def __init__(self) -> None:
        """Initialize the circuit breaker with empty quarantine sets."""
        # Modèles ayant atteint leur quota journalier
        self._exhausted_models: set[str] = set()
        # Modèles non répertoriés dans l'API Google
        self._unsupported_models: set[str] = set()

    def is_available(self, model: str) -> bool:
        """Check if a model is currently eligible for invocation.

        Args:
            model: Identifier of the Gemini model.

        Returns:
            bool: True if available, False if quarantined.
        """
        return (
            model not in self._exhausted_models
            and model not in self._unsupported_models
        )

    def filter_available_models(self, models: list[str]) -> list[str]:
        """Return only non-quarantined models preserving original priority.

        Args:
            models: Ordered candidate model cascade.

        Returns:
            list[str]: Filtered list of eligible models.
        """
        return [m for m in models if self.is_available(m)]

    def record_failure(self, model: str, error: Exception) -> bool:
        """Analyze an API error and quarantine the model if unrecoverable.

        Args:
            model: Model identifier that triggered the error.
            error: Raised exception from GenAI client.

        Returns:
            bool: True if model was quarantined (unrecoverable),
                False if transient.
        """
        err_str = str(error).lower()

        # Cas 1 : Modèle inexistant dans le catalogue Google (404)
        if any(p in err_str for p in NOT_FOUND_PATTERNS):
            self._unsupported_models.add(model)
            print(
                f"🚫 [CircuitBreaker] Modèle inexistant '{model}' (404), "
                "écarté définitivement."
            )
            return True

        # Cas 2 : Quota journalier dépassé (RPD 429)
        if any(p in err_str for p in DAILY_QUOTA_PATTERNS):
            self._exhausted_models.add(model)
            print(
                f"🛑 [CircuitBreaker] Quota journalier atteint pour '{model}', "
                "mis en quarantaine pour le reste du cycle."
            )
            return True

        # Erreur transitoire (ex: 503 ou rate limit par minute)
        return False

    def reset(self) -> None:
        """Reset all quarantined models."""
        self._exhausted_models.clear()
        self._unsupported_models.clear()


# Instance singleton globale partagée
_GLOBAL_BREAKER = ModelCircuitBreaker()


def get_circuit_breaker() -> ModelCircuitBreaker:
    """Access the shared global ModelCircuitBreaker instance.

    Returns:
        ModelCircuitBreaker: Global singleton circuit breaker.
    """
    return _GLOBAL_BREAKER
