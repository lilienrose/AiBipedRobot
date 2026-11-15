import cv2
from config import (
    ALLOWED_HAND,
    CONSENSUS_MIN_VOTES,
    HISTORY_LEN,
    MIN_GESTURE_CONFIDENCE,
    MIN_HAND_CONFIDENCE,
    MODEL_PATH,
    ROBOT_IP,
    ROBOT_PORT,
)
from hand_tracker import HandTracker
from gesture_classifier import GestureClassifier
from robot_sender import RobotSender


def run_robot_control():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker(max_hands=2, detection_conf=0.7)
    classifier = GestureClassifier(
        MODEL_PATH,
        history_len=HISTORY_LEN,
        min_confidence=MIN_GESTURE_CONFIDENCE,
        min_votes=CONSENSUS_MIN_VOTES,
    )
    sender = RobotSender(ip=ROBOT_IP, port=ROBOT_PORT)
    active_hand = None

    window_name = "Robot Gesture Control Center"
    print(f"Info: Řídicí centrum spuštěno. Robot: {ROBOT_IP}:{ROBOT_PORT}")
    print(f"Info: Povolená ruka: {ALLOWED_HAND or 'OBĚ'} | Min. jistota: {MIN_GESTURE_CONFIDENCE * 100:.0f}%")

    while True:
        success, frame = cap.read()
        if not success or frame is None or frame.size == 0:
            print("[CHYBA] Kamera nenaběhla nebo vrátila prázdný snímek.")
            break

        frame = cv2.flip(frame, 1)
        num_hands = tracker.process_frame(frame)

        if num_hands > 0:
            frame = tracker.draw_landmarks(frame)

        if num_hands > 1:
            active_hand = None
            classifier.reset()
            sender.send_command("STOP")
            text = f"STOP: Více rukou v záběru ({num_hands})!"
            color = (0, 0, 255)

        elif num_hands == 1:
            hand_label, hand_conf = tracker.get_hand_info(0)

            if ALLOWED_HAND and hand_label != ALLOWED_HAND:
                active_hand = None
                classifier.reset()
                sender.send_command("STOP")
                text = f"STOP: Nepovolená ruka ({hand_label}, vyžadována {ALLOWED_HAND})"
                color = (0, 140, 255)

            elif hand_conf < MIN_HAND_CONFIDENCE:
                text = f"Nejistá detekce ruky ({hand_conf * 100:.0f}%)..."
                color = (0, 200, 255)

            else:
                if active_hand is not None and hand_label != active_hand:
                    classifier.reset()
                active_hand = hand_label

                vector = tracker.get_normalized_vector(0)
                _, confidence, gesture_name, status = classifier.predict_smoothed(vector)

                sender.send_command(gesture_name)

                hand_cz = "PRAVÁ" if hand_label == "Right" else "LEVÁ"

                if status == "LOW_CONFIDENCE":
                    text = f"[{hand_cz}] Držím: {gesture_name} ({confidence * 100:.0f}%)"
                    color = (0, 200, 255)
                elif status == "STABILIZING":
                    text = f"[{hand_cz}] Stabilizuji gesto ({confidence * 100:.0f}%)..."
                    color = (255, 200, 0)
                else:
                    text = f"[{hand_cz}] Příkaz: {gesture_name} ({confidence * 100:.0f}%)"
                    color = (0, 255, 0)

        else:
            active_hand = None
            classifier.reset()
            sender.send_command("STOP")
            text = "Hledám ruku (STOP)..."
            color = (0, 0, 255)

        cv2.putText(
            frame,
            text,
            (20, 45),
            cv2.FONT_HERSHEY_COMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

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

    sender.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_robot_control()