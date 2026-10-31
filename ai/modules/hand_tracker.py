import os
import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'hand_landmarker.task')


base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

class HandTracker:

#---------- Class: HandTracker ------------
# Konfigurace MadiaPipe modulu
    def __init__(
        self,
        running_mode: bool = False, 
        max_hands: int = 1,
        detection_conf: float = 0.7,
        track_conf: float = 0.7,

      ):
    
     self.mode = running_mode
     self.max_hands = max_hands
     self.detenction_conf = detection_conf
     self.track_conf = track_conf

     self.mp_hands = mp.solutions.hands
     self.hands = self.mp_hands.Hands(
        static_image_mode=self.mode,
        max_num_hands=self.max_hands,
        min_detection_confidence=self.detection_conf,
        min_tracking_confidence=self.track_conf,
    )
     self.mp_draw = mp.solutions.drawing_utils

     self.results

     
#vstup celého systému, přeměňuje obraz z
#z kamery na data (X,Y,Z souřadnice)
#21 klíčových bodů ruky
def capture_cam():
    cap = cv2.VideoCapture(0)

    # Inicializace MediaPipe Hands
    tracker = HandTracker(max_hands = 1, detection_conf=0.7)

    p_time = 0

    while True:
        success, img = cap.read()
        if not success:
            print("[CHYBA] Nelze načíst obraz z kamery.")
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


    

#detection_con - jak moc si model musí být jistý, že vidí ruku

#track_con - Jak moc si musí být jistý v následujících snímcích než spustí detekci

process_frame(frame)

#Předzpracování obrazu a spuštění detekce

#Konverze barevného prostoru z BGR
#do RGB

#spuštění inferenčního modelu nad RGB snímkem

#uložení výsledku do vnitřního stavu objektu


get_landmarks(frame_width, frame_height, normalized = False)
#Extrakce souřadnic všech 21 kloubů ruky

#Pokud byla ruka detektována projde 21 kloubů

#Pokud normalized=True, vrátí relativní souřadnice

#pokud normalized = False, přepočítá je na reálné pixely obrazovky

#Return seznam nebo NumPy podle bodů

draw_landmarks(frame, draw_skeleton = True, draw_ids = False)
#Vizualizace pro debugging

#Vykreslí kostru ruky a volitelné čísla kloubů do OpenCV okna

