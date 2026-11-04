import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")

DATASET_PATH = os.path.join(DATA_DIR, "gestures_dataset.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "gesture_model.pkl")

ROBOT_IP = "192.168.1.150"  
ROBOT_PORT = 5005  

GESTURE_MAP = {
    0: "IDLE",
    1: "FIST",
    2: "OPEN_PALM",
    3: "PEACE",
    4: "THUMBS_UP",
    5: "THUMBS_DOWN",
    6: "POINTING",
}