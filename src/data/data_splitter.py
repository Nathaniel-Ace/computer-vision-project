import os
import shutil
from sklearn.model_selection import StratifiedShuffleSplit


def stratified_split_and_copy_images_by_race(df, subset_size=0.1, test_size=0.2, train_dir="../data/processed/train", test_dir="../data/processed/test"):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    subset_df = df.sample(frac=subset_size, random_state=42)

    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    for train_index, test_index in strat_split.split(subset_df, subset_df["race"]):
        train_df = subset_df.iloc[train_index]
        test_df = subset_df.iloc[test_index]

    # Kopieren der Trainingsbilder in Klassen-Unterverzeichnisse für 5 Rassenkategorien
    for _, row in train_df.iterrows():
        src_path = row['file_path']
        class_dir = os.path.join(train_dir, f"race_{row['race']}")  # Klassenverzeichnis nach Rasse
        os.makedirs(class_dir, exist_ok=True)
        dest_path = os.path.join(class_dir, os.path.basename(src_path))
        shutil.copy(src_path, dest_path)

    # Kopieren der Testbilder in Klassen-Unterverzeichnisse für 5 Rassenkategorien
    for _, row in test_df.iterrows():
        src_path = row['file_path']
        class_dir = os.path.join(test_dir, f"race_{row['race']}")  # Klassenverzeichnis nach Rasse
        os.makedirs(class_dir, exist_ok=True)
        dest_path = os.path.join(class_dir, os.path.basename(src_path))
        shutil.copy(src_path, dest_path)

    print(f"Verkleinerte, gleichmäßig nach Rasse verteilte Trainingsdaten: {len(train_df)} Bilder in {train_dir} kopiert.")
    print(f"Verkleinerte, gleichmäßig nach Rasse verteilte Testdaten: {len(test_df)} Bilder in {test_dir} kopiert.")
