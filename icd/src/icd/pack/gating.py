"""
Mode gating for pack vs RLM selection.

Determines whether to use simple pack compilation or
activate the retrieval-augmented LM (RLM) pipeline based
on query characteristics and retrieval entropy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from icd.config import Config
    from icd.retrieval.hybrid import RetrievalResult

logger = structlog.get_logger(__name__)


class RetrievalMode(str, Enum):
    """Available retrieval modes."""

    PACK = "pack"  # Simple pack compilation
    RLM = "rlm"  # Retrieval-augmented LM with iteration


@dataclass
class GatingDecision:
    """Result of mode gating decision."""

    mode: RetrievalMode
    confidence: float
    reason: str
    signals: dict[str, float]


class ModeGate:
    """
    Gate for selecting between pack and RLM modes.

    Uses multiple signals to determine the appropriate
    retrieval strategy:
    - Entropy: High entropy suggests ambiguous query
    - Score distribution: Flat distribution suggests uncertainty
    - Query complexity: Long/complex queries may need RLM
    - Contract density: Many contracts may benefit from pack
    """

    def __init__(self, config: "Config") -> None:
        """
        Initialize the mode gate.

        Args:
            config: ICD configuration.
        """
        self.config = config
        self.rlm_enabled = config.rlm.enabled
        self.entropy_threshold = config.rlm.entropy_threshold

        # Signal weights
        self.entropy_weight = 0.4
        self.score_gap_weight = 0.2
        self.query_complexity_weight = 0.2
        self.contract_density_weight = 0.2

    def decide(
        self,
        retrieval_result: "RetrievalResult",
        query: str,
    ) -> GatingDecision:
        """
        Decide which mode to use.

        Args:
            retrieval_result: Initial retrieval results.
            query: User query.

        Returns:
            GatingDecision with mode and reasoning.
        """
        if not self.rlm_enabled:
            return GatingDecision(
                mode=RetrievalMode.PACK,
                confidence=1.0,
                reason="RLM disabled in configuration",
                signals={},
            )

        # Compute signals
        signals = self._compute_signals(retrieval_result, query)

        # Compute RLM score
        rlm_score = self._compute_rlm_score(signals)

        # Make decision
        if rlm_score > 0.5:
            mode = RetrievalMode.RLM
            confidence = rlm_score
            reason = self._generate_rlm_reason(signals)
        else:
            mode = RetrievalMode.PACK
            confidence = 1.0 - rlm_score
            reason = self._generate_pack_reason(signals)

        decision = GatingDecision(
            mode=mode,
            confidence=confidence,
            reason=reason,
            signals=signals,
        )

        logger.debug(
            "Mode gate decision",
            mode=mode.value,
            confidence=confidence,
            signals=signals,
        )

        return decision

    def _compute_signals(
        self,
        result: "RetrievalResult",
        query: str,
    ) -> dict[str, float]:
        """Compute gating signals."""
        signals: dict[str, float] = {}

        # Entropy signal (normalized)
        from icd.retrieval.entropy import EntropyCalculator

        entropy_calc = EntropyCalculator()
        entropy_result = entropy_calc.compute_full(result.scores)
        signals["entropy"] = entropy_result.normalized_entropy

        # Score gap signal
        signals["score_gap"] = entropy_calc.compute_score_gap(result.scores)

        # Query complexity signal
        signals["query_complexity"] = self._compute_query_complexity(query)

        # Contract density signal
        contract_count = sum(1 for c in result.chunks if c.is_contract)
        signals["contract_density"] = (
            contract_count / len(result.chunks) if result.chunks else 0.0
        )

        # Result count signal
        signals["result_count"] = min(1.0, len(result.chunks) / 20.0)

        return signals

    def _compute_query_complexity(self, query: str) -> float:
        """
        Compute query complexity score.

        Complex queries often benefit from RLM.
        """
        score = 0.0

        # Length factor
        words = query.split()
        word_count = len(words)
        if word_count > 10:
            score += 0.3
        elif word_count > 5:
            score += 0.1

        # Question indicators
        question_words = ["how", "why", "what", "when", "where", "which", "explain"]
        if any(word.lower() in question_words for word in words):
            score += 0.2

        # Multi-part query indicators
        if " and " in query.lower() or "," in query:
            score += 0.2

        # Technical complexity
        technical_terms = [
            "implement",
            "architecture",
            "design",
            "pattern",
            "refactor",
            "optimize",
            "debug",
            "trace",
            "flow",
        ]
        if any(term in query.lower() for term in technical_terms):
            score += 0.2

        # Negation complexity
        if " not " in query.lower() or "without" in query.lower():
            score += 0.1

        return min(1.0, score)

    def _compute_rlm_score(self, signals: dict[str, float]) -> float:
        """Compute overall RLM score from signals."""
        # High entropy favors RLM
        entropy_contribution = signals.get("entropy", 0.0) * self.entropy_weight

        # Low score gap favors RLM
        gap_contribution = (
            (1.0 - signals.get("score_gap", 1.0)) * self.score_gap_weight
        )

        # High query complexity favors RLM
        complexity_contribution = (
            signals.get("query_complexity", 0.0) * self.query_complexity_weight
        )

        # Low contract density favors RLM
        contract_contribution = (
            (1.0 - signals.get("contract_density", 0.0))
            * self.contract_density_weight
        )

        return (
            entropy_contribution
            + gap_contribution
            + complexity_contribution
            + contract_contribution
        )

    def _generate_rlm_reason(self, signals: dict[str, float]) -> str:
        """Generate human-readable reason for RLM selection."""
        reasons = []

        if signals.get("entropy", 0) > 0.6:
            reasons.append("high retrieval entropy")

        if signals.get("score_gap", 1) < 0.3:
            reasons.append("similar scores across results")

        if signals.get("query_complexity", 0) > 0.5:
            reasons.append("complex query structure")

        if signals.get("contract_density", 0) < 0.2:
            reasons.append("few contract definitions")

        return "RLM selected due to: " + ", ".join(reasons) if reasons else "RLM selected"

    def _generate_pack_reason(self, signals: dict[str, float]) -> str:
        """Generate human-readable reason for pack selection."""
        reasons = []

        if signals.get("entropy", 1) < 0.4:
            reasons.append("low retrieval entropy")

        if signals.get("score_gap", 0) > 0.5:
            reasons.append("clear top results")

        if signals.get("contract_density", 0) > 0.3:
            reasons.append("high contract density")

        return "Pack selected due to: " + ", ".join(reasons) if reasons else "Pack selected"


class AdaptiveGate:
    """
    Adaptive mode gate that learns from feedback.

    Tracks decision outcomes and adjusts thresholds.
    """

    def __init__(self, config: "Config") -> None:
        """
        Initialize the adaptive gate.

        Args:
            config: ICD configuration.
        """
        self.base_gate = ModeGate(config)
        self._history: list[dict[str, Any]] = []
        self._max_history = 100

        # Adaptive thresholds
        self._entropy_threshold = config.rlm.entropy_threshold
        self._learning_rate = 0.1

    def decide(
        self,
        retrieval_result: "RetrievalResult",
        query: str,
    ) -> GatingDecision:
        """Make a gating decision."""
        decision = self.base_gate.decide(retrieval_result, query)

        # Record for learning
        self._history.append(
            {
                "signals": decision.signals.copy(),
                "mode": decision.mode,
                "query_hash": hash(query) % 10000,
            }
        )

        # Trim history
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        return decision

    def provide_feedback(
        self,
        was_successful: bool,
        user_override: RetrievalMode | None = None,
    ) -> None:
        """
        Provide feedback on the last decision.

        Args:
            was_successful: Whether the decision led to good results.
            user_override: If user manually switched modes.
        """
        if not self._history:
            return

        last_decision = self._history[-1]
        last_decision["success"] = was_successful
        last_decision["override"] = user_override

        # Adjust thresholds based on feedback
        if user_override is not None:
            signals = last_decision["signals"]
            entropy = signals.get("entropy", 0.5)

            if user_override == RetrievalMode.RLM and last_decision["mode"] == RetrievalMode.PACK:
                # User wanted RLM but we chose PACK, lower threshold
                self._entropy_threshold -= self._learning_rate * (
                    self._entropy_threshold - entropy
                )
            elif user_override == RetrievalMode.PACK and last_decision["mode"] == RetrievalMode.RLM:
                # User wanted PACK but we chose RLM, raise threshold
                self._entropy_threshold += self._learning_rate * (
                    entropy - self._entropy_threshold
                )

            # Clamp threshold
            self._entropy_threshold = max(0.3, min(0.8, self._entropy_threshold))

            logger.info(
                "Adjusted entropy threshold",
                new_threshold=self._entropy_threshold,
            )

    def get_stats(self) -> dict[str, Any]:
        """Get adaptive gate statistics."""
        if not self._history:
            return {"decisions": 0}

        pack_count = sum(1 for h in self._history if h["mode"] == RetrievalMode.PACK)
        rlm_count = len(self._history) - pack_count

        success_count = sum(1 for h in self._history if h.get("success", False))
        override_count = sum(1 for h in self._history if h.get("override") is not None)

        return {
            "decisions": len(self._history),
            "pack_count": pack_count,
            "rlm_count": rlm_count,
            "success_rate": success_count / len(self._history),
            "override_rate": override_count / len(self._history),
            "current_entropy_threshold": self._entropy_threshold,
        }
