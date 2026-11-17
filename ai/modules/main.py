import cv2
from arming_manager import ArmingManager
from config import (
    ALLOWED_HAND,
    ARM_GESTURE,
    ARM_HOLD_SECONDS,
    AUTO_DISARM_TIMEOUT,
    CONSENSUS_MIN_VOTES,
    HISTORY_LEN,
    MIN_GESTURE_CONFIDENCE,
    MIN_HAND_CONFIDENCE,
    MODEL_PATH,
    PALM_SIZE_MAX,
    PALM_SIZE_MIN,
    ROBOT_IP,
    ROBOT_PORT,
    TURN_DEADZONE,
)
from gesture_classifier import GestureClassifier
from hand_tracker import HandTracker
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
    arming = ArmingManager(
        arm_gesture=ARM_GESTURE,
        hold_seconds=ARM_HOLD_SECONDS,
        auto_disarm_timeout=AUTO_DISARM_TIMEOUT,
    )
    active_hand = None
    progress = 0.0

    window_name = "Robot Gesture Control Center"
    print(f"Info: Řídicí centrum spuštěno. Robot: {ROBOT_IP}:{ROBOT_PORT}")

    while True:
        success, frame = cap.read()
        if not success or frame is None or frame.size == 0:
            print("[CHYBA] Kamera nenaběhla nebo vrátila prázdný snímek.")
            break

        frame = cv2.flip(frame, 1)
        num_hands = tracker.process_frame(frame)

        if num_hands > 0:
            frame = tracker.draw_landmarks(frame)

        # STAV 1: Více rukou -> okamžitý STOP a bezpečnostní uzamčení
        if num_hands > 1:
            active_hand = None
            classifier.reset()
            arming.force_disarm()
            progress = 0.0
            sender.send_command("STOP")
            text = f"STOP: Více rukou ({num_hands}) -> ZAMČENO!"
            color = (0, 0, 255)

        # STAV 2: Jedna ruka v záběru
        elif num_hands == 1:
            hand_label, hand_conf = tracker.get_hand_info(0)

            if ALLOWED_HAND and hand_label != ALLOWED_HAND:
                active_hand = None
                classifier.reset()
                is_armed, progress = arming.update(None, hand_present=False)
                sender.send_command("STOP")
                text = f"STOP: Nepovolená ruka ({hand_label})"
                color = (0, 140, 255)

            elif hand_conf < MIN_HAND_CONFIDENCE:
                is_armed, progress = arming.update(None, hand_present=True)
                text = f"Nejistá detekce ruky ({hand_conf * 100:.0f}%)..."
                color = (0, 200, 255)

            else:
                if active_hand is not None and hand_label != active_hand:
                    classifier.reset()
                active_hand = hand_label

                vector = tracker.get_normalized_vector(0)
                _, confidence, gesture_name, status = classifier.predict_smoothed(vector)
                speed, turn = tracker.get_spatial_telemetry(
                    0, PALM_SIZE_MIN, PALM_SIZE_MAX, TURN_DEADZONE
                )

                valid_gesture = gesture_name if status == "OK" else None
                is_armed, progress = arming.update(valid_gesture, hand_present=True)

                hand_cz = "PRAVA" if hand_label == "Right" else "LEVA"

                if not is_armed:
                    sender.send_command("STOP")
                    text = f"[{hand_cz}] ZAMCENO | Gesto: {gesture_name} ({confidence * 100:.0f}%)"
                    color = (0, 140, 255)
                else:
                    if gesture_name == ARM_GESTURE:
                        sender.send_command("STOP")
                        text = f"[{hand_cz}] Zamykam..."
                        color = (0, 200, 255)
                    else:
                        sender.send_command(gesture_name, hand=hand_label, speed=speed, turn=turn)
                        text = (
                            f"[{hand_cz}] {gesture_name} ({confidence * 100:.0f}%) | "
                            f"Rychlost: {int(speed * 100)}% | Smer: {turn:+.2f}"
                        )
                        color = (0, 255, 0)

        else:
            active_hand = None
            classifier.reset()
            is_armed, progress = arming.update(None, hand_present=False)
            sender.send_command("STOP")
            text = "Hledam ruku (STOP)..."
            color = (0, 0, 255)

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 8), (620, 78), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv2.putText(
            frame,
            text,
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
        arming.draw_hud(frame, progress)

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