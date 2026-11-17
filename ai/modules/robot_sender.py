import json
import socket
from config import ROBOT_IP, ROBOT_PORT


class RobotSender:
    def __init__(self, ip: str = ROBOT_IP, port: int = ROBOT_PORT):
        self.ip = ip
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.last_payload: dict | None = None

    def send_command(
        self,
        command: str,
        hand: str = "NONE",
        speed: float = 0.0,
        turn: float = 0.0,
    ) -> bool:
        if not command or command == "NO_HAND":
            command = "STOP"

        if command == "STOP":
            hand = "NONE"
            speed = 0.0
            turn = 0.0

        payload = {
            "cmd": command,
            "hand": hand,
            "speed": round(speed, 2),
            "turn": round(turn, 2),
        }

        # Odesíláme jen při změně stavu nebo výraznější změně rychlosti/úhlu (>= 0.05)
        if self.last_payload is not None:
            same_cmd = payload["cmd"] == self.last_payload["cmd"]
            same_hand = payload["hand"] == self.last_payload["hand"]
            diff_speed = abs(payload["speed"] - self.last_payload["speed"])
            diff_turn = abs(payload["turn"] - self.last_payload["turn"])

            if same_cmd and same_hand and diff_speed < 0.05 and diff_turn < 0.05:
                return False

        try:
            message = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.socket.sendto(message, (self.ip, self.port))
            self.last_payload = payload
            print(f"[UDP JSON] -> {message.decode('utf-8')}")
            return True
        except Exception as e:
            print(f"[CHYBA UDP] Nepodařilo se odeslat data: {e}")
            return False

    def close(self):
        self.socket.close()