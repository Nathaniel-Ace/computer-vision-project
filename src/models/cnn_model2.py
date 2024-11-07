from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input


def build_cnn_model(input_shape=(64, 64, 3), num_classes=5):
    model = Sequential([
        Input(shape=input_shape),

        # Erste Convolutional-Block mit Dropout
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),  # 25% der Neuronen in dieser Schicht werden deaktiviert

        # Zweiter Convolutional-Block mit erhöhter Filteranzahl und Dropout
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),  # 30% Dropout für diesen Block

        # Dritter Convolutional-Block mit erhöhter Filteranzahl und Dropout
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.4),  # 40% Dropout

        # Vierter Convolutional-Block
        Conv2D(512, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.5),  # 50% Dropout

        # Flatten und vollvernetzte Dense-Schichten
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),  # 50% Dropout vor der Ausgabe

        # Ausgabe-Schicht
        Dense(num_classes, activation='softmax')
    ])
    return model
