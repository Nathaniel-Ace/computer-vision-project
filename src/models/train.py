from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


def compile_and_train(model, train_generator, test_generator, epochs=0, learning_rate=0):
    model.compile(optimizer=Adam(learning_rate=learning_rate),  # Kleinere Lernrate
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=test_generator.samples // test_generator.batch_size
    )

    return history


def evaluate_model(model, test_generator):
    loss, accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")
    return loss, accuracy


def plot_training_history(history):
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
