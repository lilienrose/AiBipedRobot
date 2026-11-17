import json
import os
import socket
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import ROBOT_PORT


def run_mock_receiver():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        sock.bind(("0.0.0.0", ROBOT_PORT))
    except OSError as e:
        print(f"[CHYBA] Port {ROBOT_PORT} je obsazený nebo nedostupný: {e}")
        print(f" Zkontroluj běžící proces příkazem: ss -ulpn | grep {ROBOT_PORT}")
        return

    print(f"[TEST UDP] Simulátor ESP32 spuštěn. Naslouchám na 0.0.0.0:{ROBOT_PORT}...")
    print("Ukončení pomocí Ctrl+C\n")

    try:
        while True:
            data, addr = sock.recvfrom(1024)
            raw_text = data.decode("utf-8", errors="replace")

            try:
                payload = json.loads(raw_text)
                cmd = payload.get("cmd", "?")
                hand = payload.get("hand", "?")
                speed = payload.get("speed", 0.0)
                turn = payload.get("turn", 0.0)
                print(
                    f"[ESP32 <- {addr[0]}] Příkaz: {cmd:<12} | Ruka: {hand:<5} | "
                    f"Rychlost: {int(speed * 100):>3}% | Směr: {turn:+.2f}"
                )
            except json.JSONDecodeError:
                print(f"[ESP32 <- {addr[0]}] Surový text: {raw_text}")

    except KeyboardInterrupt:
        print("\n[TEST UDP] Simulátor ukončen.")
    finally:
        sock.close()


if __name__ == "__main__":
    run_mock_receiver()