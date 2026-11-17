import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")

DATASET_PATH = os.path.join(DATA_DIR, "gestures_dataset.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "gesture_model.pkl")

#ROBOT_IP = "192.168.1.150"  
ROBOT_IP = "127.0.0.1"
ROBOT_PORT = 5005  

ALLOWED_HAND = None
MIN_HAND_CONFIDENCE = 0.80
MIN_GESTURE_CONFIDENCE = 0.78
HISTORY_LEN = 9
CONSENSUS_MIN_VOTES = 7

ARM_GESTURE = "PEACE" #Tímto gestem se robot odemyká a zamyká
ARM_HOLD_SECONDS = 1.5
AUTO_DISARM_TIMEOUT = 5.0

PALM_SIZE_MIN = 0.08
PALM_SIZE_MAX = 0.28
TURN_DEADZONE = 0.12

# --- OBOUSMĚRNÁ TELEMETRIE Z ESP32 ---
TELEMETRY_PORT = 5006          # Port na PC, kam ESP32 posílá data ze senzorů
TELEMETRY_TIMEOUT = 1.5        # Po kolika s bez dat označíme telemetrii jako offline
OBSTACLE_STOP_CM = 15.0        # Kritická vzdálenost HC-SR04 (blokuje pohyb vpřed)
FALL_ANGLE_DEG = 42.0          # Kritický náklon Pitch/Roll z MPU6050 (pád robota)
FORWARD_GESTURES = {"OPEN_PALM", "POINTING"}  # Gesta pro chůzi vpřed

# --- DETEKCE NEZNÁMÝCH GEST (OOD ENTROPIE) ---
MAX_PREDICTION_ENTROPY = 0.85  # Maximální povolená Shannonova entropie rozdělení

GESTURE_MAP = {
    0: "IDLE",
    1: "FIST",
    2: "OPEN_PALM",
    3: "PEACE",
    4: "THUMBS_UP",
    5: "THUMBS_DOWN",
    6: "POINTING",
}