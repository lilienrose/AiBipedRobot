from collections import Counter, deque
import os
import cv2
import joblib
import numpy as np

from config import GESTURE_MAP, MODEL_PATH
from hand_tracker import HandTracker


class GestureClassifier:

    def __init__(self, model_path: str, history_len: int = 7):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Chyba: Model nenalezen na cestě: {model_path}. Nejprve spusť train_model.py!"
            )
        self.model = joblib.load(model_path)
        self.history = deque(maxlen=history_len)
        
#Zkontroluje, zda vektor není None a zda má 63 hdonot
#Přizpůsobí tvar matice na 2D pole
#Vypočte predikci a jistotu
#Vrátí dvojici (ID_GESTA, confidence)
    def predict(self, vector: np.ndarray | None) -> tuple[int | None, float]:
        """Vrátí okamžitou predikci gesta (ID) a jeho jistotu (0.0 - 1.0)."""
        if vector is None or len(vector) != 63:
            return None, 0.0

        input_data = vector.reshape(1, -1)

        prediction = int(self.model.predict(input_data)[0])
        probabilities = self.model.predict_proba(input_data)[0]
        confidence = float(np.max(probabilities))

        return prediction, confidence

#Získá surový odhad ze self.predict (ID gesta a jistotu)
#Pokud ruka není v obraze staré příkazy se smažou
#aby robot po ukázání ruky nepřehrával staré příkazy
#Spočítá se nejčastější ID pro poslední snímky
#ID nejčastějšího gesta se vyhledá ve slovníku
    def predict_smoothed(
        self, vector: np.ndarray | None
    ) -> tuple[int | None, float, str]:
        raw_label, confidence = self.predict(vector)

        if raw_label is None:
            self.history.clear()
            return None, 0.0, "NO_HAND"

        self.history.append(raw_label)

        most_common_label = Counter(self.history).most_common(1)[0][0]
        gesture_name = GESTURE_MAP.get(most_common_label, "UNKNOWN")

        return most_common_label, confidence, gesture_name


#Otevře kameru, připraví detektor načte HandTracker
#načte vytrénovaný model
#Získá 63D vektor, s procentem jistoty
#vykreslí zjištěné gesto. 
def run_classifier():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker(max_hands=1, detection_conf=0.7)
    classifier = GestureClassifier(MODEL_PATH, history_len=7)
    window_name = "Real-Time Gesture Recognition"

    while True:
        success, frame = cap.read()
        if not success:
            print("[CHYBA] Kamera nenaběhla.")
            break

        frame = cv2.flip(frame, 1)
        has_hand = tracker.process_frame(frame)

        if has_hand:
            frame = tracker.draw_landmarks(frame)
            vector = tracker.get_normalized_vector()
            _, confidence, name = classifier.predict_smoothed(vector)
            text, color = f"Gesto: {name} ({confidence * 100:.0f}%)", (0, 255, 0)
        else:
            classifier.predict_smoothed(None)
            text, color = "Hledam ruku...", (0, 0, 255)

        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        try:
            if key == ord("q") or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_classifier()

    

    

