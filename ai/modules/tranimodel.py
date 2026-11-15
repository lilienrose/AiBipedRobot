import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


from config import DATASET_PATH, GESTURE_MAP, MODEL_PATH

#Načítání .csv datasetu, trénování klasifikačního modelu
#uloží vytrénovaný model do souboru .pk


#Zkontroluje, zda soubor s datasetem existuje
#Načte CSV soubor numpy.loadtxt
#Spráruje 0.sloupec jako štítky gest (y) a 
#zbývajících 63 sloupců jako souřadnice kloubů (x)
#ověří, že nečtená data nemají nulový počet vzorků
def load_dataset(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Nenalezen dataset na: {filepath}")
    data = np.loadtxt(filepath, delimiter = ",", skiprows = 1)

    if data.size == 0:
        raise ValueError("Dataset je prázdný.")

    if data.ndim == 1:
        data = np.expand_dims(data, axis = 0)

    y = data[:, 0].astype(int)
    x = data[:, 1:]

    print (
        f"Dataset načten úspěšně: {x.shape[0]} vzorků, {x.shape[1]} příznaků."
    )
    return x,y
def augment_with_rotations(
    x: np.ndarray, y: np.ndarray, angles_deg: tuple[int, ...] = (-25, -12, 12, 25)
) -> tuple[np.ndarray, np.ndarray]:
    """Rozšíří trénovací data o pootočené varianty ruky (řeší náklon ruky bez nového sběru)."""
    augmented_x = [x]
    augmented_y = [y]

    coords = x.reshape(-1, 21, 3)

    for deg in angles_deg:
        rad = np.deg2rad(deg)
        cos_a, sin_a = np.cos(rad), np.sin(rad)

        # 1. Rotace kolem osy Z (náklon ruky doleva/doprava před kamerou)
        rot_z = np.array([
            [cos_a, -sin_a, 0.0],
            [sin_a,  cos_a, 0.0],
            [0.0,    0.0,   1.0],
        ])
        rotated_z = (coords @ rot_z.T).reshape(-1, 63)
        augmented_x.append(rotated_z)
        augmented_y.append(y)

        # 2. Rotace kolem osy Y (natočení dlaně mírně bokem)
        rot_y = np.array([
            [ cos_a, 0.0, sin_a],
            [ 0.0,   1.0, 0.0  ],
            [-sin_a, 0.0, cos_a],
        ])
        rotated_y = (coords @ rot_y.T).reshape(-1, 63)
        augmented_x.append(rotated_y)
        augmented_y.append(y)

    x_out = np.vstack(augmented_x)
    y_out = np.concatenate(augmented_y)
    print(f"Po augmentaci rotacemi: {x_out.shape[0]} trénovacích vzorků.")
    return x_out, y_out
#Rozdělí dataset 80% pro trénink a 20% testování
#Použije Random Forestse 100 rozhodovacími stromy
#Otestuje vytrénovaný model na 20% testovacích dat
#Vypočítá přenost a zobrazí detailní přehled pro každé gesto
def train_and_evaluate(x: np.ndarray, y: np.ndarray):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    x_train_aug, y_train_aug = augment_with_rotations(x_train, y_train)

    model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(x_train_aug, y_train_aug)

    accuracy = model.score(x_test, y_test)
    print(f"Přesnost modelu na reálných testovacích datech: {accuracy * 100:.2f} %")
    return model

def save_model(model: RandomForestClassifier, filepath: str) -> None:
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    joblib.dump(model, filepath)
    print(f"Model je úspěšně uložen do: {filepath}")

def main():
    print("=== TRÉNINK ML MODELU PRO REPOZITÁŘ ROBOTA ===")
    X, y = load_dataset(DATASET_PATH)
    model = train_and_evaluate(X, y)
    save_model(model, MODEL_PATH)

if __name__ == "__main__":
    main()
