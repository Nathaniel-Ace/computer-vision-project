# src/data/dataset_creator.py

import os
import pandas as pd
from src.data.label_extractor import extract_labels_from_filename  # Importieren der Extraktionsfunktion

def create_dataset(dataset_path, save_path="../data/processed", csv_filename="dataset.csv"):
    """
    Erstellt ein Dataset aus Bildern und ihren Labels, die aus den Dateinamen extrahiert werden,
    und speichert es als CSV-Datei.

    Args:
        dataset_path (str): Pfad zum Verzeichnis mit den Bildern.
        save_path (str): Pfad zum Verzeichnis, in dem das CSV gespeichert wird (Standard: ../data/processed).
        csv_filename (str): Name der CSV-Datei (Standard: dataset.csv).

    Returns:
        pd.DataFrame: DataFrame mit Dateipfaden und zugehörigen Labels (Alter, Geschlecht, Rasse).
    """
    data = []
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg"):
            file_path = os.path.join(dataset_path, filename)
            file_path = file_path.replace("\\", "/")  # Umwandlung zu Vorwärts-Slashes
            print(f"Verarbeite Datei: {file_path}")  # Debug-Ausgabe
            try:
                # Labels aus dem Dateinamen extrahieren
                labels = extract_labels_from_filename(filename)
                if labels:  # Nur gültige Labels hinzufügen
                    age, gender, race = labels
                    data.append((file_path, age, gender, race))
            except ValueError as e:
                print(f"Fehler beim Verarbeiten der Datei {filename}: {e}")

    # Erstellen eines DataFrames
    df = pd.DataFrame(data, columns=["file_path", "age", "gender", "race"])

    # Speichern als CSV-Datei im angegebenen Pfad
    os.makedirs(save_path, exist_ok=True)
    csv_path = os.path.join(save_path, csv_filename).replace("\\", "/")  # Sicherstellen, dass der CSV-Pfad Vorwärts-Slashes verwendet
    df.to_csv(csv_path, index=False)
    print(f"Dataset gespeichert unter: {csv_path}")

    return df
