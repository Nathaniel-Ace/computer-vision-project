# src/data/label_extractor.py
def extract_labels_from_filename(filename):
    # Remove the file extension
    filename = filename.split('.')[0]

    # Split the filename by underscores
    parts = filename.split('_')

    # Check if the filename has the expected number of parts
    if len(parts) < 3:
        raise ValueError(f"Invalid file format: {filename}")

    try:
        age = int(parts[0])
        gender = int(parts[1])
        race = int(parts[2])
    except ValueError as e:
        raise ValueError(f"Error parsing labels from filename: {filename}") from e

    return age, gender, race