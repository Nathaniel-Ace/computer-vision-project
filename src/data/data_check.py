# src/data/data_check.py

import os
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed


def check_duplicates(dataset_path):
    """
    Überprüft den Ordner auf Duplikate basierend auf Dateinamen.

    Args:
        dataset_path (str): Pfad zum Verzeichnis mit den Bildern.

    Returns:
        list: Liste der Duplikate.
    """
    file_names = set()
    duplicates = []

    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg"):
            if filename in file_names:
                duplicates.append(filename)
            else:
                file_names.add(filename)

    return duplicates


def check_file_integrity(file_path):
    """
    Überprüft, ob eine Datei leer oder beschädigt ist.

    Args:
        file_path (str): Der vollständige Pfad zur Datei.

    Returns:
        str: Der Dateiname, falls die Datei beschädigt ist, andernfalls None.
    """
    image = cv2.imread(file_path)
    if image is None:
        return os.path.basename(file_path)
    return None


def check_corrupted_files(dataset_path, max_workers=8):
    """
    Überprüft den Ordner auf beschädigte oder leere Dateien mit paralleler Verarbeitung.

    Args:
        dataset_path (str): Pfad zum Verzeichnis mit den Bildern.
        max_workers (int): Anzahl der parallelen Threads (Standard ist 8).

    Returns:
        list: Liste der beschädigten oder leeren Dateien.
    """
    corrupted_files = []
    file_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".jpg")]

    # Verwenden Sie ThreadPoolExecutor für parallele Verarbeitung
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks and collect results
        futures = {executor.submit(check_file_integrity, path): path for path in file_paths}

        for future in as_completed(futures):
            result = future.result()
            if result:  # Wenn das Ergebnis nicht None ist, ist die Datei beschädigt
                corrupted_files.append(result)

    return corrupted_files
