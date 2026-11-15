from collections import Counter, deque
import os
import joblib
import numpy as np

from config import (
    CONSENSUS_MIN_VOTES,
    GESTURE_MAP,
    HISTORY_LEN,
    MIN_GESTURE_CONFIDENCE,
)


class GestureClassifier:
    def __init__(
        self,
        model_path: str,
        history_len: int = HISTORY_LEN,
        min_confidence: float = MIN_GESTURE_CONFIDENCE,
        min_votes: int = CONSENSUS_MIN_VOTES,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Chyba: Model nenalezen na cestě: {model_path}. Nejprve spusť trainmodel.py!"
            )
        self.model = joblib.load(model_path)
        self.history = deque(maxlen=history_len)
        self.min_confidence = min_confidence
        self.min_votes = min_votes
        self.last_confirmed_label: int | None = None

    def reset(self) -> None:
        """Okamžitě vymaže historii a vrátí stav na bezpečný klid (STOP)."""
        self.history.clear()
        self.last_confirmed_label = None

    def predict(self, vector: np.ndarray | None) -> tuple[int | None, float]:
        """Vrátí okamžitou predikci gesta (ID) a jeho jistotu (0.0 - 1.0)."""
        if vector is None or len(vector) != 63:
            return None, 0.0

        input_data = vector.reshape(1, -1)
        probabilities = self.model.predict_proba(input_data)[0]
        best_idx = int(np.argmax(probabilities))
        prediction = int(self.model.classes_[best_idx])
        confidence = float(probabilities[best_idx])

        return prediction, confidence

    def predict_smoothed(
        self, vector: np.ndarray | None
    ) -> tuple[int | None, float, str, str]:
        """
        Vrátí: (potvrzené_ID, aktuální_jistota, název_příkazu, stavová_zpráva)
        """
        if vector is None:
            self.reset()
            return None, 0.0, "STOP", "NO_HAND"

        raw_label, confidence = self.predict(vector)

        # 1. Ochrana podle procent jistoty: nejistý snímek (při náklonu) ignorujeme
        if raw_label is None or confidence < self.min_confidence:
            current_name = (
                GESTURE_MAP.get(self.last_confirmed_label, "STOP")
                if self.last_confirmed_label is not None
                else "STOP"
            )
            return self.last_confirmed_label, confidence, current_name, "LOW_CONFIDENCE"

        # 2. Přidání spolehlivého snímku do historie
        self.history.append(raw_label)

        # 3. Hlasování většinou (hystereze)
        most_common_label, count = Counter(self.history).most_common(1)[0]
        if count >= self.min_votes:
            self.last_confirmed_label = most_common_label

        if self.last_confirmed_label is None:
            return None, confidence, "STOP", "STABILIZING"

        gesture_name = GESTURE_MAP.get(self.last_confirmed_label, "STOP")
        return self.last_confirmed_label, confidence, gesture_name, "OK"