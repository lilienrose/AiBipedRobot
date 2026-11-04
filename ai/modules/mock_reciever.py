import socket
from config import ROBOT_PORT


def run_mock_receiver():
    # Posloucháme na všech síťových rozhraních PC
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", ROBOT_PORT))

    print(
        f"[TEST UDP] Simulátor ESP32 spuštěn. Naslouchám na portu {ROBOT_PORT}..."
    )

    while True:
        data, addr = sock.recvfrom(1024)
        print(f"[ESP32 PŘIJATO od {addr[0]}]: {data.decode('utf-8')}")


if __name__ == "__main__":
    run_mock_receiver()