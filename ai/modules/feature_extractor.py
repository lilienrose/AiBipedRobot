import numpy as np

# Trojice kloubů (A, B, C) pro výpočet kosinu úhlu v kloubu B
FINGER_ANGLE_TRIPLETS = (
    (1, 2, 3), (2, 3, 4),       # Palec
    (0, 5, 6), (5, 6, 7),       # Ukazováček
    (0, 9, 10), (9, 10, 11),    # Prostředníček
    (0, 13, 14), (13, 14, 15),  # Prsteníček
    (0, 17, 18), (17, 18, 19),  # Malíček
)

FINGERTIP_INDICES = (4, 8, 12, 16, 20)


def extract_features_from_coords(coords_21x3: np.ndarray) -> np.ndarray:
    """
    Z pole tvaru (21, 3) vytvoří 78-prvkový příznakový vektor:
    - 63 normalizovaných souřadnic (posun do zápěstí, škálování velikostí dlaně)
    - 10 kosinů úhlů ohybu prstů (100% invariantní vůči 3D rotaci ruky)
    - 5 vzdáleností špiček prstů od zápěstí
    """
    wrist = coords_21x3[0]
    translated = coords_21x3 - wrist

    palm_size = float(np.linalg.norm(translated[9]))
    if palm_size < 1e-6:
        palm_size = 1.0

    norm_coords = translated / palm_size

    angles = []
    for a, b, c in FINGER_ANGLE_TRIPLETS:
        v1 = norm_coords[a] - norm_coords[b]
        v2 = norm_coords[c] - norm_coords[b]
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
        cos_angle = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
        angles.append(cos_angle)

    tip_distances = [
        float(np.linalg.norm(norm_coords[tip_idx])) for tip_idx in FINGERTIP_INDICES
    ]

    return np.concatenate([
        norm_coords.flatten(),
        np.array(angles, dtype=np.float32),
        np.array(tip_distances, dtype=np.float32),
    ])


def expand_dataset_features(x_63: np.ndarray) -> np.ndarray:
    """Převede matici (N, 63) ze stávajícího CSV datasetu na matici (N, 78)."""
    coords_batch = x_63.reshape(-1, 21, 3)
    return np.vstack([extract_features_from_coords(c) for c in coords_batch])