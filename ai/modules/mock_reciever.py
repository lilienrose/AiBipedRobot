import json
import math
import os
import socket
import sys
import time
import cv2
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import ROBOT_PORT


class RealisticBipedSimulator:
    def __init__(self, width: int = 1040, height: int = 640):
        self.w = width
        self.h = height
        self.arena_w = 740

        self.cmd = "STOP"
        self.mode_label = "KLID / STOP (90 deg)"
        self.hand = "-"
        self.speed = 0.0
        self.turn = 0.0
        self.last_packet_time = 0.0

        self.pos_x = self.arena_w / 2.0
        self.pos_y = height / 2.0 + 40.0
        self.heading = -math.pi / 2
        self.trail: list[tuple[int, int]] = [(int(self.pos_x), int(self.pos_y))]

        self.servos = {
            "L_HIP": 90.0,
            "R_HIP": 90.0,
            "L_FOOT": 90.0,
            "R_FOOT": 90.0,
            "L_ARM": 90.0,
            "R_ARM": 90.0,
        }

        self.phase = 0.0
        self.last_frame_time = time.time()

    def update_from_payload(self, payload: dict | str) -> None:
        if isinstance(payload, dict):
            self.cmd = str(payload.get("cmd", "STOP")).strip().upper()
            self.hand = str(payload.get("hand", "-"))
            self.speed = float(payload.get("speed", 0.0))
            self.turn = float(payload.get("turn", 0.0))
        else:
            self.cmd = str(payload).strip().upper()
            if self.cmd in ("POINTING_LEFT", "TURN_LEFT"):
                self.turn = -0.7
            elif self.cmd in ("POINTING_RIGHT", "TURN_RIGHT"):
                self.turn = 0.7
        self.last_packet_time = time.time()

    def _classify_intent(self) -> tuple[str, float, str]:
        if self.cmd in ("STOP", "IDLE", "HOME", "DISARMED"):
            return "STOP", 0.0, "KLID / STOP (90 deg)"

        if self.cmd == "PEACE":
            return "STOP", 0.0, "ZAMKNUTO / PAUZA (90 deg)"

        if self.cmd in ("OPEN_PALM", "FORWARD"):
            return "WALK_FWD", 0.0, "PLYNULA CHUZE VPRED"

        if self.cmd in ("FIST", "BACKWARD"):
            return "WALK_BACK", 0.0, "COUVANI"

        if self.cmd in ("POINTING_LEFT", "TURN_LEFT"):
            return "TURN", -0.75, "ZATACENI VLEVO"

        if self.cmd in ("POINTING_RIGHT", "TURN_RIGHT"):
            return "TURN", 0.75, "ZATACENI VPRAVO"

        if self.cmd == "POINTING":
            if self.turn < -0.15:
                return "TURN", self.turn, "ZATACENI VLEVO"
            if self.turn > 0.15:
                return "TURN", self.turn, "ZATACENI VPRAVO"
            return "STOP", 0.0, "UKAZANI (STRED - CEKAM)"

        if self.cmd == "THUMBS_UP":
            return "SQUAT", 0.0, "PODREP (BALANCNI POZA)"

        if self.cmd == "THUMBS_DOWN":
            return "TIPTOE", 0.0, "VZPRIMENI NA SPICKY"

        return "STOP", 0.0, "KLID / STOP (90 deg)"

    def step_physics(self) -> None:
        now = time.time()
        dt = min(now - self.last_frame_time, 0.05)
        self.last_frame_time = now

        if now - self.last_packet_time > 0.5:
            self.cmd = "IDLE"
            self.speed = 0.0
            self.turn = 0.0

        mode, turn_dir, self.mode_label = self._classify_intent()
        is_walking = mode in ("WALK_FWD", "WALK_BACK", "TURN")
        eff_speed = max(self.speed, 0.5) if is_walking else 0.0

        targets = {
            "L_HIP": 90.0,
            "R_HIP": 90.0,
            "L_FOOT": 90.0,
            "R_FOOT": 90.0,
            "L_ARM": 90.0,
            "R_ARM": 90.0,
        }

        if is_walking:
            step_freq = 4.5 + eff_speed * 6.0
            self.phase = (self.phase + dt * step_freq) % (2.0 * math.pi)
            hip_wave = math.sin(self.phase)
            tilt_wave = math.cos(self.phase)

            dir_sign = -1.0 if mode == "WALK_BACK" else 1.0
            amp_l = 15.0 * dir_sign * (1.0 + turn_dir * 0.5)
            amp_r = 15.0 * dir_sign * (1.0 - turn_dir * 0.5)

            targets["L_HIP"] = 90.0 + hip_wave * amp_l
            targets["R_HIP"] = 90.0 - hip_wave * amp_r
            targets["L_FOOT"] = 90.0 + tilt_wave * 15.0
            targets["R_FOOT"] = 90.0 + tilt_wave * 15.0
            targets["L_ARM"] = 90.0 - hip_wave * (20.0 * dir_sign)
            targets["R_ARM"] = 90.0 + hip_wave * (20.0 * dir_sign)

            step_pulse = 0.75 + 0.25 * abs(math.cos(self.phase))
            if mode == "WALK_FWD":
                forward_vel = (95.0 * eff_speed) * step_pulse
                self.pos_x += math.cos(self.heading) * forward_vel * dt
                self.pos_y += math.sin(self.heading) * forward_vel * dt
            elif mode == "WALK_BACK":
                backward_vel = -(75.0 * eff_speed) * step_pulse
                self.pos_x += math.cos(self.heading) * backward_vel * dt
                self.pos_y += math.sin(self.heading) * backward_vel * dt
            else:
                angular_vel = turn_dir * 1.9 * step_pulse
                forward_vel = 35.0 * step_pulse
                self.heading += angular_vel * dt
                self.pos_x += math.cos(self.heading) * forward_vel * dt
                self.pos_y += math.sin(self.heading) * forward_vel * dt

        elif mode == "SQUAT":
            targets["L_FOOT"] = 70.0
            targets["R_FOOT"] = 70.0
            targets["L_ARM"] = 110.0
            targets["R_ARM"] = 110.0

        elif mode == "TIPTOE":
            targets["L_FOOT"] = 105.0
            targets["R_FOOT"] = 105.0
            targets["L_ARM"] = 75.0
            targets["R_ARM"] = 75.0

        alpha = min(1.0, dt * 12.0)
        for key in self.servos:
            self.servos[key] += (targets[key] - self.servos[key]) * alpha

        self.pos_x = max(65.0, min(self.arena_w - 65.0, self.pos_x))
        self.pos_y = max(125.0, min(self.h - 55.0, self.pos_y))

        pt = (int(self.pos_x), int(self.pos_y))
        if not self.trail or math.hypot(pt[0] - self.trail[-1][0], pt[1] - self.trail[-1][1]) > 5:
            self.trail.append(pt)
            if len(self.trail) > 300:
                self.trail.pop(0)
    def _rot_offset(self, forward: float, right: float) -> tuple[int, int]:
        fx, fy = math.cos(self.heading), math.sin(self.heading)
        rx, ry = -fy, fx
        return (
            int(self.pos_x + fx * forward + rx * right),
            int(self.pos_y + fy * forward + ry * right),
        )

    def draw_arena(self, canvas: np.ndarray) -> None:
        for x in range(0, self.arena_w, 50):
            cv2.line(canvas, (x, 0), (x, self.h), (34, 37, 44), 1)
        for y in range(0, self.h, 50):
            cv2.line(canvas, (0, y), (self.arena_w, y), (34, 37, 44), 1)

        cv2.rectangle(canvas, (35, 92), (self.arena_w - 35, self.h - 30), (75, 80, 92), 2)

        for i in range(1, len(self.trail)):
            cv2.line(canvas, self.trail[i - 1], self.trail[i], (0, 195, 240), 2, cv2.LINE_AA)

    def draw_robot(self, canvas: np.ndarray) -> None:
        l_hip_deg = self.servos["L_HIP"] - 90.0
        r_hip_deg = self.servos["R_HIP"] - 90.0
        l_foot_deg = self.servos["L_FOOT"] - 90.0
        r_foot_deg = self.servos["R_FOOT"] - 90.0

        tilt_deg = (l_foot_deg + r_foot_deg) * 0.5
        l_lift = max(0.0, -l_foot_deg) * 0.45
        r_lift = max(0.0, r_foot_deg) * 0.45
        body_sway = -tilt_deg * 0.35

        # Kladná hodnota = podřep (SQUAT), záporná = vzpřímení na špičky (TIPTOE)
        # Vychází z L_ARM/R_ARM: 110° -> +20 (podřep), 75° -> -15 (špičky)
        vertical_pose = self.servos["L_ARM"] - 90.0 if self.servos["L_ARM"] == self.servos["R_ARM"] else 0.0
        squat_offset = vertical_pose * 0.45
        stance_spread = max(-4.0, vertical_pose * 0.25)  # Rozkročení v podřepu, zúžení na špičkách

        # Stín pod robotem (větší v podřepu, menší při vzpřímení)
        shadow_radius = int(max(24, min(42, 34 + vertical_pose * 0.3)))
        cv2.circle(canvas, self._rot_offset(0, body_sway), shadow_radius, (16, 18, 22), -1)

        # Chodidla
        for side_sign, hip_deg, lift in ((-1, l_hip_deg, l_lift), (1, r_hip_deg, r_lift)):
            stride_px = hip_deg * 0.85
            foot_lat = side_sign * (17.0 + stance_spread) + body_sway
            foot_center = self._rot_offset(stride_px, foot_lat)
            foot_ang = self.heading + math.radians(hip_deg * 0.35)
            fx, fy = math.cos(foot_ang), math.sin(foot_ang)

            # Při vzpřímení na špičkách (vertical_pose < -5) se chodidlo zkrátí (opora jen o špičku)
            foot_len_back = 6 if vertical_pose < -5 else 12
            p_back = (int(foot_center[0] - fx * foot_len_back), int(foot_center[1] - fy * foot_len_back))
            p_front = (int(foot_center[0] + fx * 15), int(foot_center[1] + fy * 15))

            is_grounded = lift < 2.0 and vertical_pose >= -5
            color = (0, 235, 160) if is_grounded else ((255, 180, 60) if vertical_pose < -5 else (120, 130, 145))
            cv2.line(canvas, p_back, p_front, color, int(13 + lift * 0.3), cv2.LINE_AA)
            cv2.circle(canvas, foot_center, 4, (30, 30, 35), -1)

        # Ruce (v podřepu vyvažují dopředu, na špičkách se roztáhnou do stran)
        for side_sign, servo_key in ((-1, "L_ARM"), (1, "R_ARM")):
            arm_deg = self.servos[servo_key] - 90.0
            arm_fwd = arm_deg * 0.95
            arm_out = 30.0 + abs(arm_deg) * 0.35
            shoulder = self._rot_offset(0, side_sign * 24 + body_sway)
            hand = self._rot_offset(arm_fwd, side_sign * arm_out + body_sway)
            cv2.line(canvas, shoulder, hand, (55, 58, 66), 8, cv2.LINE_AA)
            cv2.circle(canvas, hand, 7, (230, 210, 40), -1)

        # Trup a hlava (výška se výrazně mění mezi podřepem a špičkami)
        torso_base = self._rot_offset(0, body_sway)
        torso_top = (torso_base[0], int(torso_base[1] - 26 + squat_offset))
        cv2.circle(canvas, torso_base, 24, (42, 44, 52), -1)
        cv2.circle(canvas, torso_top, 23, (58, 62, 72), -1)
        cv2.circle(canvas, torso_top, 23, (220, 200, 50), 2)

        head_center = (torso_top[0], int(torso_top[1] - 18 + squat_offset * 0.3))
        cv2.circle(canvas, head_center, 18, (45, 48, 56), -1)
        cv2.circle(canvas, head_center, 18, (220, 200, 50), 2)

        # Oči ultrazvuku HC-SR04
        fx, fy = math.cos(self.heading), math.sin(self.heading)
        rx, ry = -fy, fx
        for eye_side in (-7, 7):
            ex = int(head_center[0] + fx * 13 + rx * eye_side)
            ey = int(head_center[1] + fy * 13 + ry * eye_side)
            cv2.circle(canvas, (ex, ey), 4, (240, 240, 240), -1)
            cv2.circle(canvas, (ex, ey), 2, (20, 20, 25), -1)

        # Směrová šipka
        cv2.arrowedLine(
            canvas, self._rot_offset(28, 0), self._rot_offset(52, 0), (0, 255, 140), 2, tipLength=0.35
        )

    def draw_servo_panel(self, canvas: np.ndarray) -> None:
        px = self.arena_w + 10
        pw = self.w - px - 15
        cv2.rectangle(canvas, (px, 12), (px + pw, self.h - 30), (30, 33, 40), -1)
        cv2.rectangle(canvas, (px, 12), (px + pw, self.h - 30), (85, 90, 105), 1)

        cv2.putText(canvas, "TELEMETRIE 6x SG90", (px + 18, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 230, 240), 2)
        cv2.putText(canvas, "Stred kalibrace = 90 deg", (px + 18, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 165, 175), 1)

        labels = [
            ("L_HIP  (Leva kycle)", "L_HIP"),
            ("R_HIP  (Prava kycle)", "R_HIP"),
            ("L_FOOT (Levy kotnik)", "L_FOOT"),
            ("R_FOOT (Pravy kotnik)", "R_FOOT"),
            ("L_ARM  (Leva ruka)", "L_ARM"),
            ("R_ARM  (Prava ruka)", "R_ARM"),
        ]

        y = 115
        for title, key in labels:
            angle = self.servos[key]
            cv2.putText(canvas, f"{title}: {angle:5.1f} deg", (px + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 230), 1)

            bar_x, bar_y, bar_w, bar_h = px + 18, y + 10, pw - 36, 14
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 54, 64), -1)

            mid_x = bar_x + bar_w // 2
            cv2.line(canvas, (mid_x, bar_y - 3), (mid_x, bar_y + bar_h + 3), (255, 255, 255), 1)

            pos_ratio = max(0.0, min(1.0, angle / 180.0))
            cur_x = int(bar_x + pos_ratio * bar_w)
            cv2.line(canvas, (mid_x, bar_y + bar_h // 2), (cur_x, bar_y + bar_h // 2), (0, 220, 255), 6)
            cv2.circle(canvas, (cur_x, bar_y + bar_h // 2), 6, (0, 255, 150), -1)

            y += 72

        cv2.circle(canvas, (px + 25, self.h - 75), 6, (0, 235, 160), -1)
        cv2.putText(canvas, "Chodidlo na zemi (taha)", (px + 40, self.h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 210), 1)
        cv2.circle(canvas, (px + 25, self.h - 50), 6, (120, 130, 145), -1)
        cv2.putText(canvas, "Chodidlo ve vzduchu", (px + 40, self.h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 210), 1)

    def render(self) -> np.ndarray:
        canvas = np.full((self.h, self.w, 3), (24, 26, 32), dtype=np.uint8)
        self.draw_arena(canvas)
        self.draw_robot(canvas)
        self.draw_servo_panel(canvas)

        cv2.rectangle(canvas, (15, 12), (self.arena_w - 15, 80), (18, 20, 26), -1)
        cv2.rectangle(canvas, (15, 12), (self.arena_w - 15, 80), (90, 95, 110), 1)

        cmd_color = (0, 120, 255) if self.mode_label.startswith(("KLID", "ZAMKNUTO")) else (0, 255, 120)
        cv2.putText(
            canvas,
            f"GESTO: {self.cmd}  ->  {self.mode_label}",
            (30, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            cmd_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Ruka: {self.hand}  |  Rychlost: {int(self.speed * 100)}%  |  Pozice/Turn: {self.turn:+.2f}",
            (30, 69),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (220, 220, 230),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "[R] Reset | [Q] Konec",
            (self.arena_w - 185, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (160, 165, 175),
            1,
            cv2.LINE_AA,
        )
        return canvas


def run_visual_mock_receiver():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setblocking(False)

    try:
        sock.bind(("0.0.0.0", ROBOT_PORT))
    except OSError as e:
        print(f"[CHYBA] Port {ROBOT_PORT} je obsazený: {e}")
        return

    sim = RealisticBipedSimulator()
    window_name = "Robot Lee - Fyzikalni Simulator Bipeda"
    print(f"[SIMULATOR] Naslouchám na UDP portu {ROBOT_PORT}...")

    try:
        while True:
            while True:
                try:
                    data, _ = sock.recvfrom(1024)
                    raw_msg = data.decode("utf-8", errors="replace").strip()
                    try:
                        payload = json.loads(raw_msg)
                    except json.JSONDecodeError:
                        payload = raw_msg
                    sim.update_from_payload(payload)
                except BlockingIOError:
                    break

            sim.step_physics()
            frame = sim.render()
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(16) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                sim.pos_x = sim.arena_w / 2.0
                sim.pos_y = sim.h / 2.0 + 40.0
                sim.heading = -math.pi / 2
                sim.trail.clear()

            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

    finally:
        sock.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_visual_mock_receiver()