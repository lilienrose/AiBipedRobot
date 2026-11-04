import socket
from config import ROBOT_IP, ROBOT_PORT


class RobotSender:

    def __init__(self, ip: str = ROBOT_IP, port: int = ROBOT_PORT):
        self.ip = ip
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.last_sent_command = None

    def send_command(
        self, command: str, send_only_on_change: bool = True
    ) -> bool:
        """Odešle textový příkaz přes UDP na ESP32.

        Pokud je send_only_on_change=True, odesílá data pouze při změně gesta,
        aby se zbytečně nezahlcovala síť.
        """
        if not command or command == "NO_HAND":
            command = "STOP"

        if send_only_on_change and command == self.last_sent_command:
            return False

        try:
            message = command.encode("utf-8")
            self.socket.sendto(message, (self.ip, self.port))

            self.last_sent_command = command
            print(f"[UDP SENT] -> {self.ip}:{self.port} | Příkaz: {command}")
            return True
        except Exception as e:
            print(f"[CHYBA UDP] Nepodařilo se odeslat data: {e}")
            return False

    def close(self):
        self.socket.close()