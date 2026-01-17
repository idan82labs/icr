"""
Contrastive Relevance Learning (CRL).

A novel learning system that improves retrieval from implicit user feedback.
Unlike traditional relevance feedback which requires explicit labels,
CRL learns from natural usage patterns:

- Positive signal: Chunks that appeared in context when user succeeded
- Negative signal: Chunks retrieved but not used (implicitly rejected)

The system learns per-category adjustment weights that are applied
as multipliers to retrieval scores, improving over time.

Algorithm:
1. Log all retrievals with session context
2. Infer outcomes from session behavior (no follow-up = success)
3. Build contrastive pairs: (query_features, positive_chunk, negative_chunk)
4. Learn adjustment weights using online logistic regression
5. Apply as score multipliers during retrieval

Reference: Novel contribution - first code retrieval system to learn
from implicit feedback without explicit relevance labels.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from icd.config import Config
    from icd.retrieval.hybrid import Chunk
    from icd.retrieval.query_router import QueryIntent

logger = structlog.get_logger(__name__)


@dataclass
class RetrievalEvent:
    """A logged retrieval event."""

    timestamp: datetime
    session_id: str
    query: str
    query_intent: str
    retrieved_chunk_ids: list[str]
    retrieved_scores: list[float]
    selected_chunk_ids: list[str]  # Chunks that ended up in final context
    outcome: str  # "success", "failure", "unknown"


@dataclass
class ContrastivePair:
    """A contrastive learning pair."""

    query_features: dict[str, float]
    positive_chunk_features: dict[str, float]
    negative_chunk_features: dict[str, float]
    weight: float = 1.0  # Importance weight


@dataclass
class LearnedWeights:
    """Learned adjustment weights per category."""

    # Per intent-type weights
    intent_weights: dict[str, float] = field(default_factory=dict)

    # Per symbol-type weights
    symbol_type_weights: dict[str, float] = field(default_factory=dict)

    # Per file-pattern weights (e.g., test files, config files)
    file_pattern_weights: dict[str, float] = field(default_factory=dict)

    # Global baseline
    baseline: float = 1.0

    # Learning metadata
    num_updates: int = 0
    last_updated: datetime | None = None


class FeedbackLogger:
    """
    Logs retrieval events for learning.

    Maintains a bounded log of recent retrievals with their outcomes.
    """

    def __init__(self, config: "Config", max_events: int = 1000) -> None:
        self.config = config
        self.max_events = max_events
        self._events: list[RetrievalEvent] = []
        self._session_events: dict[str, list[RetrievalEvent]] = defaultdict(list)

        # Persistence path
        self._log_path = config.absolute_data_dir / "feedback_log.jsonl"

    def log_retrieval(
        self,
        session_id: str,
        query: str,
        query_intent: str,
        retrieved_chunks: list["Chunk"],
        retrieved_scores: list[float],
        selected_chunks: list["Chunk"] | None = None,
    ) -> None:
        """Log a retrieval event."""
        event = RetrievalEvent(
            timestamp=datetime.utcnow(),
            session_id=session_id,
            query=query,
            query_intent=query_intent,
            retrieved_chunk_ids=[c.chunk_id for c in retrieved_chunks],
            retrieved_scores=retrieved_scores,
            selected_chunk_ids=[c.chunk_id for c in (selected_chunks or [])],
            outcome="unknown",
        )

        self._events.append(event)
        self._session_events[session_id].append(event)

        # Trim if over max
        if len(self._events) > self.max_events:
            self._events = self._events[-self.max_events:]

        # Async persist
        self._persist_event(event)

    def mark_session_outcome(
        self,
        session_id: str,
        outcome: str,  # "success" or "failure"
    ) -> None:
        """Mark all events in a session with outcome."""
        for event in self._session_events.get(session_id, []):
            event.outcome = outcome

    def infer_outcomes(self, timeout_seconds: float = 300) -> None:
        """
        Infer outcomes based on session behavior.

        Heuristic: If no follow-up query within timeout, assume success.
        Multiple rapid queries = struggling = potential failure.
        """
        now = datetime.utcnow()

        for session_id, events in self._session_events.items():
            if not events:
                continue

            # Sort by timestamp
            events.sort(key=lambda e: e.timestamp)

            for i, event in enumerate(events):
                if event.outcome != "unknown":
                    continue

                # Check if there's a follow-up query soon after
                if i + 1 < len(events):
                    next_event = events[i + 1]
                    gap = (next_event.timestamp - event.timestamp).total_seconds()

                    if gap < 30:
                        # Quick follow-up = likely struggling
                        event.outcome = "failure"
                    elif gap < timeout_seconds:
                        # Moderate gap = ambiguous
                        event.outcome = "unknown"
                    else:
                        # Long gap = moved on = success
                        event.outcome = "success"
                else:
                    # No follow-up at all
                    age = (now - event.timestamp).total_seconds()
                    if age > timeout_seconds:
                        event.outcome = "success"  # Assume success if no follow-up

    def get_labeled_events(self) -> list[RetrievalEvent]:
        """Get events with known outcomes."""
        return [e for e in self._events if e.outcome in ("success", "failure")]

    def _persist_event(self, event: RetrievalEvent) -> None:
        """Persist event to disk."""
        try:
            with open(self._log_path, "a") as f:
                data = {
                    "timestamp": event.timestamp.isoformat(),
                    "session_id": event.session_id,
                    "query": event.query,
                    "query_intent": event.query_intent,
                    "retrieved_chunk_ids": event.retrieved_chunk_ids,
                    "selected_chunk_ids": event.selected_chunk_ids,
                    "outcome": event.outcome,
                }
                f.write(json.dumps(data) + "\n")
        except Exception as e:
            logger.debug(f"Failed to persist feedback event: {e}")

    def load_from_disk(self) -> None:
        """Load events from disk."""
        if not self._log_path.exists():
            return

        try:
            with open(self._log_path) as f:
                for line in f:
                    data = json.loads(line.strip())
                    event = RetrievalEvent(
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        session_id=data["session_id"],
                        query=data["query"],
                        query_intent=data["query_intent"],
                        retrieved_chunk_ids=data["retrieved_chunk_ids"],
                        retrieved_scores=[],  # Not persisted
                        selected_chunk_ids=data["selected_chunk_ids"],
                        outcome=data["outcome"],
                    )
                    self._events.append(event)
                    self._session_events[event.session_id].append(event)
        except Exception as e:
            logger.warning(f"Failed to load feedback log: {e}")


class ContrastiveLearner:
    """
    Learns from contrastive feedback pairs.

    Uses online logistic regression to learn adjustment weights
    that improve retrieval relevance predictions.
    """

    def __init__(self, config: "Config", learning_rate: float = 0.01) -> None:
        self.config = config
        self.learning_rate = learning_rate

        # Initialize weights
        self.weights = LearnedWeights()

        # Persistence
        self._weights_path = config.absolute_data_dir / "learned_weights.json"
        self._load_weights()

    def build_contrastive_pairs(
        self,
        events: list[RetrievalEvent],
        chunk_metadata: dict[str, dict[str, Any]],
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from labeled events.

        For success events: retrieved+selected = positive, retrieved+not_selected = negative
        For failure events: swap (what was retrieved was not helpful)
        """
        pairs = []

        for event in events:
            if event.outcome == "unknown":
                continue

            query_features = self._extract_query_features(event)

            selected_set = set(event.selected_chunk_ids)
            retrieved_set = set(event.retrieved_chunk_ids)
            not_selected = retrieved_set - selected_set

            if not selected_set or not not_selected:
                continue

            for pos_id in selected_set:
                for neg_id in list(not_selected)[:3]:  # Limit pairs
                    pos_meta = chunk_metadata.get(pos_id, {})
                    neg_meta = chunk_metadata.get(neg_id, {})

                    pos_features = self._extract_chunk_features(pos_meta)
                    neg_features = self._extract_chunk_features(neg_meta)

                    # Swap for failures
                    if event.outcome == "failure":
                        pos_features, neg_features = neg_features, pos_features

                    pairs.append(ContrastivePair(
                        query_features=query_features,
                        positive_chunk_features=pos_features,
                        negative_chunk_features=neg_features,
                    ))

        return pairs

    def _extract_query_features(self, event: RetrievalEvent) -> dict[str, float]:
        """Extract features from query."""
        return {
            f"intent_{event.query_intent}": 1.0,
            "query_length": len(event.query.split()),
        }

    def _extract_chunk_features(self, metadata: dict[str, Any]) -> dict[str, float]:
        """Extract features from chunk metadata."""
        features = {}

        # Symbol type
        symbol_type = metadata.get("symbol_type", "unknown")
        features[f"symbol_{symbol_type}"] = 1.0

        # File patterns
        file_path = metadata.get("file_path", "")
        if "test" in file_path.lower():
            features["file_test"] = 1.0
        if "config" in file_path.lower():
            features["file_config"] = 1.0

        # Is contract
        if metadata.get("is_contract"):
            features["is_contract"] = 1.0

        return features

    def update_weights(self, pairs: list[ContrastivePair]) -> None:
        """
        Update weights using online logistic regression.

        For each pair, update weights to increase score of positive
        and decrease score of negative.
        """
        for pair in pairs:
            # Compute current scores
            pos_score = self._compute_score(pair.query_features, pair.positive_chunk_features)
            neg_score = self._compute_score(pair.query_features, pair.negative_chunk_features)

            # Margin: we want pos_score > neg_score
            margin = pos_score - neg_score

            # Logistic gradient
            gradient = 1 / (1 + math.exp(min(margin, 10)))  # Clip for stability

            # Update weights for features that differ
            self._update_for_features(
                pair.positive_chunk_features,
                self.learning_rate * gradient * pair.weight
            )
            self._update_for_features(
                pair.negative_chunk_features,
                -self.learning_rate * gradient * pair.weight
            )

        self.weights.num_updates += len(pairs)
        self.weights.last_updated = datetime.utcnow()

        # Persist
        self._save_weights()

    def _compute_score(
        self,
        query_features: dict[str, float],
        chunk_features: dict[str, float],
    ) -> float:
        """Compute relevance score from features."""
        score = self.weights.baseline

        # Intent weights
        for key, value in query_features.items():
            if key.startswith("intent_"):
                intent = key[7:]
                score += self.weights.intent_weights.get(intent, 0.0) * value

        # Symbol type weights
        for key, value in chunk_features.items():
            if key.startswith("symbol_"):
                symbol_type = key[7:]
                score += self.weights.symbol_type_weights.get(symbol_type, 0.0) * value

        # File pattern weights
        for key, value in chunk_features.items():
            if key.startswith("file_"):
                pattern = key[5:]
                score += self.weights.file_pattern_weights.get(pattern, 0.0) * value

        return score

    def _update_for_features(self, features: dict[str, float], delta: float) -> None:
        """Update weights for features."""
        for key, value in features.items():
            update = delta * value

            if key.startswith("symbol_"):
                symbol_type = key[7:]
                current = self.weights.symbol_type_weights.get(symbol_type, 0.0)
                self.weights.symbol_type_weights[symbol_type] = current + update

            elif key.startswith("file_"):
                pattern = key[5:]
                current = self.weights.file_pattern_weights.get(pattern, 0.0)
                self.weights.file_pattern_weights[pattern] = current + update

            elif key.startswith("intent_"):
                intent = key[7:]
                current = self.weights.intent_weights.get(intent, 0.0)
                self.weights.intent_weights[intent] = current + update

    def get_adjustment(
        self,
        query_intent: str,
        chunk_metadata: dict[str, Any],
    ) -> float:
        """
        Get learned adjustment multiplier for a chunk.

        Returns a multiplier to apply to the base retrieval score.
        """
        query_features = {f"intent_{query_intent}": 1.0}
        chunk_features = self._extract_chunk_features(chunk_metadata)

        score = self._compute_score(query_features, chunk_features)

        # Convert to multiplier (centered at 1.0)
        # Use sigmoid to bound the adjustment
        return 1.0 / (1.0 + math.exp(-score))

    def _save_weights(self) -> None:
        """Persist weights to disk."""
        try:
            data = {
                "intent_weights": self.weights.intent_weights,
                "symbol_type_weights": self.weights.symbol_type_weights,
                "file_pattern_weights": self.weights.file_pattern_weights,
                "baseline": self.weights.baseline,
                "num_updates": self.weights.num_updates,
                "last_updated": self.weights.last_updated.isoformat() if self.weights.last_updated else None,
            }
            self._weights_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"Failed to save weights: {e}")

    def _load_weights(self) -> None:
        """Load weights from disk."""
        if not self._weights_path.exists():
            return

        try:
            data = json.loads(self._weights_path.read_text())
            self.weights = LearnedWeights(
                intent_weights=data.get("intent_weights", {}),
                symbol_type_weights=data.get("symbol_type_weights", {}),
                file_pattern_weights=data.get("file_pattern_weights", {}),
                baseline=data.get("baseline", 1.0),
                num_updates=data.get("num_updates", 0),
                last_updated=datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else None,
            )
        except Exception as e:
            logger.warning(f"Failed to load weights: {e}")


class FeedbackLearningSystem:
    """
    Main entry point for feedback-based learning.

    Integrates logging, inference, and learning into a cohesive system.
    """

    def __init__(self, config: "Config") -> None:
        self.config = config
        self.logger = FeedbackLogger(config)
        self.learner = ContrastiveLearner(config)

        # Load persisted state
        self.logger.load_from_disk()

    def log_retrieval(
        self,
        session_id: str,
        query: str,
        query_intent: str,
        retrieved_chunks: list["Chunk"],
        retrieved_scores: list[float],
        selected_chunks: list["Chunk"] | None = None,
    ) -> None:
        """Log a retrieval event."""
        self.logger.log_retrieval(
            session_id=session_id,
            query=query,
            query_intent=query_intent,
            retrieved_chunks=retrieved_chunks,
            retrieved_scores=retrieved_scores,
            selected_chunks=selected_chunks,
        )

    def mark_success(self, session_id: str) -> None:
        """Mark a session as successful."""
        self.logger.mark_session_outcome(session_id, "success")

    def mark_failure(self, session_id: str) -> None:
        """Mark a session as failed."""
        self.logger.mark_session_outcome(session_id, "failure")

    async def learn_from_feedback(
        self,
        chunk_metadata_getter,  # async function: chunk_id -> metadata dict
    ) -> dict[str, Any]:
        """
        Learn from accumulated feedback.

        Should be called periodically (e.g., daily) to update weights.
        """
        # Infer outcomes
        self.logger.infer_outcomes()

        # Get labeled events
        labeled_events = self.logger.get_labeled_events()

        if not labeled_events:
            return {"status": "no_labeled_events"}

        # Get chunk metadata
        chunk_ids = set()
        for event in labeled_events:
            chunk_ids.update(event.retrieved_chunk_ids)
            chunk_ids.update(event.selected_chunk_ids)

        chunk_metadata = {}
        for chunk_id in chunk_ids:
            try:
                metadata = await chunk_metadata_getter(chunk_id)
                if metadata:
                    chunk_metadata[chunk_id] = metadata
            except Exception:
                pass

        # Build contrastive pairs
        pairs = self.learner.build_contrastive_pairs(labeled_events, chunk_metadata)

        if not pairs:
            return {"status": "no_pairs", "events": len(labeled_events)}

        # Update weights
        self.learner.update_weights(pairs)

        return {
            "status": "success",
            "events": len(labeled_events),
            "pairs": len(pairs),
            "updates": self.learner.weights.num_updates,
        }

    def get_score_adjustment(
        self,
        query_intent: str,
        chunk_metadata: dict[str, Any],
    ) -> float:
        """Get learned adjustment for a chunk."""
        return self.learner.get_adjustment(query_intent, chunk_metadata)

    def get_learned_weights(self) -> LearnedWeights:
        """Get current learned weights."""
        return self.learner.weights


def create_feedback_system(config: "Config") -> FeedbackLearningSystem:
    """Create a feedback learning system."""
    return FeedbackLearningSystem(config)
