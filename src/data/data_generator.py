from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_generators(train_dir, test_dir, batch_size=64):
    # Datenaugmentation für den Trainingsgenerator
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,  # Normalisierung
        rotation_range=20,  # Zufällige Drehungen um bis zu 20 Grad
        width_shift_range=0.2,  # Zufällige horizontale Verschiebung um bis zu 20%
        height_shift_range=0.2,  # Zufällige vertikale Verschiebung um bis zu 20%
        shear_range=0.2,  # Schertransformation
        zoom_range=0.2,  # Zufälliges Zoomen
        horizontal_flip=True,  # Zufällige horizontale Spiegelung
        brightness_range = [0.5, 1.5]
    )

    # Generator ohne Augmentation für den Testdatensatz
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Trainingsdaten-Generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Testdaten-Generator
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, test_generator
