import json
import math
import socket
import threading
import time
import cv2
import numpy as np

from config import (
    FALL_ANGLE_DEG,
    OBSTACLE_STOP_CM,
    TELEMETRY_PORT,
    TELEMETRY_TIMEOUT,
)


class TelemetryReceiver:
    def __init__(self, port: int = TELEMETRY_PORT):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", self.port))
        self.sock.settimeout(0.5)

        self._lock = threading.Lock()
        self._running = True
        self.dist_cm: float | None = None
        self.pitch: float = 0.0
        self.roll: float = 0.0
        self.last_rx_time: float = 0.0

        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def _listen_loop(self) -> None:
        while self._running:
            try:
                data, _ = self.sock.recvfrom(512)
                payload = json.loads(data.decode("utf-8", errors="ignore"))
                with self._lock:
                    if "dist_cm" in payload:
                        self.dist_cm = float(payload["dist_cm"])
                    if "pitch" in payload:
                        self.pitch = float(payload["pitch"])
                    if "roll" in payload:
                        self.roll = float(payload["roll"])
                    self.last_rx_time = time.time()
            except socket.timeout:
                continue
            except Exception:
                continue

    def get_state(self) -> dict:
        """Vrátí aktuální telemetrii a vyhodnocené bezpečnostní příznaky."""
        with self._lock:
            is_online = (time.time() - self.last_rx_time) <= TELEMETRY_TIMEOUT
            dist = self.dist_cm if is_online else None
            pitch = self.pitch if is_online else 0.0
            roll = self.roll if is_online else 0.0

        obstacle_blocked = dist is not None and 0.0 < dist < OBSTACLE_STOP_CM
        robot_fallen = is_online and (
            abs(pitch) > FALL_ANGLE_DEG or abs(roll) > FALL_ANGLE_DEG
        )

        return {
            "online": is_online,
            "dist_cm": dist,
            "pitch": pitch,
            "roll": roll,
            "obstacle_blocked": obstacle_blocked,
            "robot_fallen": robot_fallen,
        }

    def draw_cockpit_hud(self, frame: np.ndarray) -> None:
        """Vykreslí radar překážek (HC-SR04) a umělý horizont (MPU6050) vpravo dole."""
        state = self.get_state()
        h, w, _ = frame.shape

        box_w, box_h = 250, 125
        x0, y0 = w - box_w - 15, h - box_h - 15

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (100, 100, 100), 1)

        if not state["online"]:
            cv2.putText(
                frame, "ESP32 TELEMETRIE: OFFLINE", (x0 + 12, y0 + 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (140, 140, 140), 1, cv2.LINE_AA,
            )
            return

        # --- 1. LEVÁ ČÁST: RADAR PŘEKÁŽEK (HC-SR04) ---
        dist = state["dist_cm"]
        bar_x, bar_y, bar_w, bar_h = x0 + 18, y0 + 28, 26, 78
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)

        if dist is not None:
            ratio = float(np.clip(dist / 100.0, 0.0, 1.0))
            fill_h = int(bar_h * ratio)
            if state["obstacle_blocked"]:
                r_color = (0, 0, 255)
            elif dist < 35.0:
                r_color = (0, 200, 255)
            else:
                r_color = (0, 255, 0)

            cv2.rectangle(
                frame, (bar_x, bar_y + bar_h - fill_h),
                (bar_x + bar_w, bar_y + bar_h), r_color, -1,
            )
            cv2.putText(
                frame, f"{dist:.0f}cm", (x0 + 10, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, r_color, 1, cv2.LINE_AA,
            )

        # --- 2. PRAVÁ ČÁST: UMĚLÝ HORIZONT (MPU6050) ---
        cx, cy = x0 + 160, y0 + 62
        radius = 36
        cv2.circle(frame, (cx, cy), radius, (90, 90, 90), 1, cv2.LINE_AA)

        # Pevný kříž referenční roviny kamery
        cv2.line(frame, (cx - 12, cy), (cx + 12, cy), (200, 200, 200), 1)
        cv2.line(frame, (cx, cy - 8), (cx, cy + 8), (200, 200, 200), 1)

        # Pohyblivá čára náklonu (Roll = úhel čáry, Pitch = svislý posun)
        roll_rad = math.radians(state["roll"])
        pitch_offset = int(np.clip(state["pitch"], -30.0, 30.0) * 0.8)
        dx = int(math.cos(roll_rad) * (radius - 4))
        dy = int(math.sin(roll_rad) * (radius - 4))

        h_color = (0, 0, 255) if state["robot_fallen"] else (0, 255, 255)
        cv2.line(
            frame,
            (cx - dx, cy + pitch_offset - dy),
            (cx + dx, cy + pitch_offset + dy),
            h_color,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"P:{state['pitch']:+.0f} R:{state['roll']:+.0f}",
            (x0 + 105, y0 + 114),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            h_color,
            1,
            cv2.LINE_AA,
        )

    def close(self) -> None:
        self._running = False
        self.sock.close()