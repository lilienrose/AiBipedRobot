import csv
import os
import time
import cv2
import numpy as np
from hand_tracker import HandTracker



from config import DATASET_PATH, GESTURE_MAP, MODEL_PATH

GESTURE_MAP = {
    0: "IDLE",
    1: "FIST",
    2: "OPEN_PALM",
    3: "PEACE",
    4: "THUMBS_UP",
    5: "THUMBS_DOWN",
    6: "POINTING",
}


class DatasetWriter:
    def __init__(self, filepath: str):
        self.filepath = filepath
        directory = os.path.dirname(self.filepath)

        if directory:
            os.makedirs(directory, exist_ok=True)

        self._ensure_header()

    def _ensure_header(self) -> None:
        """Zapíše hlavičku CSV (label + 63 souřadnic), pokud soubor ještě neexistuje."""
        if not os.path.exists(self.filepath):
            header = ["label"]
            # Vygeneruje názvy sloupců: x0, y0, z0, x1, y1, z1 ... x20, y20, z20
            for i in range(21):
                header.extend([f"x{i}", f"y{i}", f"z{i}"])

            with open(self.filepath, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)

    # validuje, že vektor není none, před tím, než se pokusí zapisovat na disk
    # a že obsahuje 63 prvků (21 kloubů x 3 souřadnice)
    # Převádí NumPy pole na Python list
    # na konec csv zapíše nové řádky
    # Potvrdí uložení
    def save_sample(self, label: int, vector: np.ndarray | None) -> bool:
        if vector is None or len(vector) != 63:
            return False

        row = [label] + vector.tolist()

        with open(self.filepath, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        return True

    # Vytvoří slovník, kde jsou klíče (ID gest)
    # Zkontroluje zda csv existuje
    # Otevře soubor csv v modu read
    # Z prvního sloupce načte label a přičte +1 ke gestu
    # ve slovníku
    # vrátí hotový slovník
    def get_counts(self, gesture_map: dict[int, str]) -> dict[int, int]:
        counts = {gid: 0 for gid in gesture_map.keys()}

        if not os.path.exists(self.filepath):
            return counts

        with open(self.filepath, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)

            for row in reader:
                if row:
                    try:
                        label = int(row[0])
                        if label in counts:
                            counts[label] += 1
                    except (ValueError, IndexError):
                        continue

        return counts


# V levém rohu vytvoří tmavý obdelník aby byl
# bílý text čitelný bez ohledu na pozadí místnosti
# projde gesture_map dict, zobrazí ID gesta, název, počet vzorků
# Pokud je dané gesto vybrané, řádek se obarví zeleně
# Neaktivní gesta zůstanou šedá
# Indikátorstavu REC/Manual
def draw_hud(
    frame: np.ndarray,
    counts: dict[int, int],
    gesture_map: dict[int, str],
    active_label: int | None,
    recording: bool,
) -> None:

    roi = frame[10:260, 10:290]
    cv2.addWeighted(roi, 0.4, np.zeros_like(roi), 0.6, 0, roi)

    # 2. Nadpis
    cv2.putText(
        frame,
        "DATASET COLLECTOR",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )

    y = 58
    for gid, name in gesture_map.items():
        active = active_label == gid
        color = (0, 255, 0) if active else (200, 200, 200)
        prefix = ">" if active else " "
        text = f"{prefix} [{gid}] {name}: {counts.get(gid, 0)}"

        cv2.putText(
            frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
        )
        y += 22

    is_rec = recording and active_label is not None
    status = (
        f"REC: [{gesture_map.get(active_label, '')}]"
        if is_rec
        else "MODE: Manual (Press 0-6)"
    )
    color = (0, 0, 255) if is_rec else (0, 255, 255)

    cv2.putText(
        frame, status, (20, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
    )



#Hlavní řídící loop sběru dat. 
#Propojuje vstup z kamery, detekci ruky, zápis vektoru co CSV
#a reakce na klávesnici
def run_collector():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker(max_hands=1, detection_conf=0.7)
    writer = DatasetWriter(DATASET_PATH)

    counts = writer.get_counts(GESTURE_MAP)
    active_label = None
    recording = False
    window_name = "Gesture Data Collector"

    while True:
        success, frame = cap.read()
        if not success:
            print("[CHYBA] Kamera nenaběhla.")
            break

        frame = cv2.flip(frame, 1)
        has_hand = tracker.process_frame(frame)
        vector = None

        if has_hand:
            frame = tracker.draw_landmarks(frame)
            vector = tracker.get_normalized_vector()

            if recording and active_label is not None:
                if writer.save_sample(active_label, vector):
                    counts[active_label] += 1

        # Vykreslení HUD a okna
        draw_hud(frame, counts, GESTURE_MAP, active_label, recording)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF

        try:
            if (
                key == ord("q")
                or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
            ):
                break
        except cv2.error:
            break

        if ord("0") <= key <= ord("6"):
            selected_gid = key - ord("0")
            if selected_gid in GESTURE_MAP:
                active_label = selected_gid
                if not recording and has_hand:
                    if writer.save_sample(active_label, vector):
                        counts[active_label] += 1

        elif key == 32 and active_label is not None:
            recording = not recording

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_collector()