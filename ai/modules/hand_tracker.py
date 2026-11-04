import os
import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

os.environ["QT_QPA_PLATFORM"] = "xcb"
if "DISPLAY" not in os.environ:
    os.environ["DISPLAY"] = ":0"


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'hand_landmarker.task')
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),        # Palec
    (0, 5), (5, 6), (6, 7), (7, 8),        # Ukazováček
    (5, 9), (9, 10), (10, 11), (11, 12),   # Prostředníček
    (9, 13), (13, 14), (14, 15), (15, 16), # Prsteníček
    (13, 17), (17, 18), (18, 19), (19, 20),# Malíček
    (0, 17)                                # Dlaň
)


class HandTracker:

#---------- Class: HandTracker --------

#inicializuje detektor MediaPipe Hand Landmarker s konfigurací
    def __init__(
        self,
        max_hands: int = 1,
        detection_conf: float = 0.7,
        track_conf: float = 0.7,
    ):
        self.max_hands = max_hands
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=self.max_hands,
            min_hand_detection_confidence=detection_conf,
            min_tracking_confidence=track_conf,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.results = None

     
 #Příjme snímek z OpenCV, převede jej do formutu pro mp
 #Spustí detekci ruky
    def process_frame(self, frame: np.ndarray) -> bool:
    
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data = frame_rgb)

        self.results = self.detector.detect(mp_image)

        return bool(self.results and self.results.hand_landmarks)


# Převádí souřadnice 21 kloubů detekované ruky
#na normalizovaný 63-prvkový vektor(21 x 3)
#který není závislý na poloze ruky v obraze
#nebo její vzdálenosti od kamery
    def get_normalized_vector(self) -> np.ndarray | None:
        if not self.results or not self.results.hand_landmarks:
            return None

        detected_hand = self.results.hand_landmarks[0]
        coords = np.array([[lm.x, lm.y, lm.z]for lm in detected_hand])

        wrist = coords[0]

        translated = coords - wrist

        palm_size = np.linalg.norm(translated[9])
        if palm_size < 1e-6:
            palm_size = 1.0

        return ( translated / palm_size).flatten()
                               
                               
#Vykresluje vizuální kostru ruky, kloubky
#volitelně jejich ID indexy přímo do snímku videa
    def draw_landmarks(
    self, frame: np.ndarray, draw_ids: bool = False
) -> np.ndarray:
        if not self.results or not self.results.hand_landmarks:
            return frame

        h, w, _ = frame.shape
        for hand in self.results.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

     
            for p1, p2 in HAND_CONNECTIONS:
                cv2.line(frame, pts[p1], pts[p2], (0, 255, 0), 2)

            for i, pt in enumerate(pts):
                color = (
                (255, 0, 255) if i in (4, 8, 12, 16, 20) else (0, 0, 255)
            )  
                cv2.circle(frame, pt, 6, color, -1)

                if draw_ids:
                    cv2.putText(
                    frame,
                    str(i),
                    (pt[0] + 5, pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 0),
                    1,
                )

        return frame
    
#Hlavní testovací smyčka pro spuštění hand_tracker.py modulu
def capture_cam():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker(max_hands = 1, detection_conf=0.7)

    p_time = 0

    while True:
        success, img = cap.read()
        if not success:
            print("Chyba: Nelze načíst obraz z kamery.")
            break

        img = cv2.flip(img, 1)
        
        captured_hand = tracker.process_frame(img)

        if captured_hand:
            img = tracker.draw_landmarks(img)
            vector = tracker.get_normalized_vector()
        

        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time

        cv2.putText(
            img,
            f"FPS: {int(fps)}",
            (10, 70),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            (255, 0, 255),
            3,
        )

        cv2.imshow("Hand Tracking Preview", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    capture_cam()

