from collections import Counter, deque
import os
import joblib
import numpy as np

from config import (
    CONSENSUS_MIN_VOTES,
    GESTURE_MAP,
    HISTORY_LEN,
    MAX_PREDICTION_ENTROPY,
    MIN_GESTURE_CONFIDENCE,
)


class GestureClassifier:
    def __init__(
        self,
        model_path: str,
        history_len: int = HISTORY_LEN,
        min_confidence: float = MIN_GESTURE_CONFIDENCE,
        min_votes: int = CONSENSUS_MIN_VOTES,
        max_entropy: float = MAX_PREDICTION_ENTROPY,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Chyba: Model nenalezen na cestě: {model_path}. Nejprve spusť trainmodel.py!"
            )
        self.model = joblib.load(model_path)
        self.history = deque(maxlen=history_len)
        self.min_confidence = min_confidence
        self.min_votes = min_votes
        self.max_entropy = max_entropy
        self.last_confirmed_label: int | None = None

    def reset(self) -> None:
        """Okamžitě vymaže historii a vrátí stav na bezpečný klid (STOP)."""
        self.history.clear()
        self.last_confirmed_label = None

    def predict(self, vector: np.ndarray | None) -> tuple[int | None, float, float]:
        if vector is None or len(vector) != 78:
            return None, 0.0, 99.0

        input_data = vector.reshape(1, -1)
        probabilities = self.model.predict_proba(input_data)[0]

        best_idx = int(np.argmax(probabilities))
        prediction = int(self.model.classes_[best_idx])
        confidence = float(probabilities[best_idx])

        probs_safe = np.clip(probabilities, 1e-12, 1.0)
        entropy = float(-np.sum(probs_safe * np.log(probs_safe)))

        return prediction, confidence, entropy

    def predict_smoothed(
        self, vector: np.ndarray | None
    ) -> tuple[int | None, float, str, str]:
        if vector is None:
            self.reset()
            return None, 0.0, "STOP", "NO_HAND"

        raw_label, confidence, entropy = self.predict(vector)

        if raw_label is None or entropy > self.max_entropy:
            current_name = (
                GESTURE_MAP.get(self.last_confirmed_label, "STOP")
                if self.last_confirmed_label is not None
                else "STOP"
            )
            return self.last_confirmed_label, confidence, current_name, "UNKNOWN_OOD"

        if confidence < self.min_confidence:
            current_name = (
                GESTURE_MAP.get(self.last_confirmed_label, "STOP")
                if self.last_confirmed_label is not None
                else "STOP"
            )
            return self.last_confirmed_label, confidence, current_name, "LOW_CONFIDENCE"

        self.history.append(raw_label)

        most_common_label, count = Counter(self.history).most_common(1)[0]
        if count >= self.min_votes:
            self.last_confirmed_label = most_common_label

        if self.last_confirmed_label is None:
            return None, confidence, "STOP", "STABILIZING"

        gesture_name = GESTURE_MAP.get(self.last_confirmed_label, "STOP")
        return self.last_confirmed_label, confidence, gesture_name, "OK"