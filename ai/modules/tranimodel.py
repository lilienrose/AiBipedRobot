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
        f"Dataset načten úspěšně: {X.shape[0]} vzorků, {X.shape[1]} příznaků."
    )
    return x,y

#Rozdělí dataset 80% pro trénink a 20% testování
#Použije Random Forestse 100 rozhodovacími stromy
#Otestuje vytrénovaný model na 20% testovacích dat
#Vypočítá přenost a zobrazí detailní přehled pro každé gesto
def train_and_evaluate(x: np.ndarray, y: np.ndarray):
    x_train, x_test, y_train, y_test = train_test_split(
        x,y,test_size=0.2, random_state=42, stratify = y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)

    print(f"Přenost modelu je: {accuracy * 100:.2f} %")
    return model 

        