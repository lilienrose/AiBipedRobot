import cv2
from config import GESTURE_MAP, MODEL_PATH, ROBOT_IP, ROBOT_PORT
from hand_tracker import HandTracker
from gesture_classifier import GestureClassifier
from robot_sender import RobotSender


def run_robot_control():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker(max_hands=1, detection_conf=0.7)
    classifier = GestureClassifier(MODEL_PATH, history_len=7)
    sender = RobotSender(ip=ROBOT_IP, port=ROBOT_PORT)

    window_name = "Robot Gesture Control Center"
    print(f"Info: Řídicí centrum spuštěno. Robot: {ROBOT_IP}:{ROBOT_PORT}")
    print("info: Ukončení klávesou 'q'.")

    while True:
        success, frame = cap.read()
        if not success or frame is None or frame.size == 0:
            print("[CHYBA] Kamera nenaběhla nebo vrátila prázdný snímek.")
            break

        frame = cv2.flip(frame, 1)
        has_hand = tracker.process_frame(frame)

        if has_hand:
            frame = tracker.draw_landmarks(frame)
            vector = tracker.get_normalized_vector()

            _, confidence, gesture_name = classifier.predict_smoothed(vector)

            sender.send_command(gesture_name)

            text = f"Příkaz: {gesture_name} ({confidence * 100:.0f}%)"
            color = (0, 255, 0)
        else:
            classifier.predict_smoothed(None)
            sender.send_command("STOP")

            text = "Hledám ruku (STOP)..."
            color = (0, 0, 255)

        cv2.putText(
            frame,
            text,
            (20, 50),
            cv2.FONT_HERSHEY_COMPLEX,
            0.8,
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