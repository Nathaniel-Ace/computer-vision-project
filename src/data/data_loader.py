# src/data/data_loader.py

import os
import tensorflow as tf


def parse_image(filename):
    """
    Parses an image filename to extract the age label and load the image.

    Args:
        filename (str): Path to the image file.

    Returns:
        tuple: A tuple containing the image tensor and the age label.
    """
    try:
        # Extract age from filename (first part before '_')
        age = tf.strings.to_number(tf.strings.split(filename, "_")[0], out_type=tf.int32)

        # Load and preprocess image
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=3)  # Decode JPEG
        image = tf.image.resize(image, [128, 128])  # Resize to (128, 128)
        image = image / 255.0  # Normalize pixel values to [0, 1]

        return image, age
    except tf.errors.InvalidArgumentError:
        print(f"Warning: Skipping file due to parsing error: {filename}")
        return None, None


def load_dataset(dataset_path, batch_size=32, augment=False):
    filepaths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".jpg")]

    # Create a dataset from file paths
    dataset = tf.data.Dataset.from_tensor_slices(filepaths)
    dataset = dataset.map(lambda x: parse_image(x), num_parallel_calls=tf.data.AUTOTUNE)

    # Filter out any failed parses
    dataset = dataset.filter(lambda image, age: image is not None and age is not None)

    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    if augment:
        dataset = dataset.map(lambda x, y: augment_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)

    return dataset


def augment_image(image, label):
    """
    Applies data augmentation to an image.

    Args:
        image (tf.Tensor): The image tensor.
        label (int): The age label associated with the image.

    Returns:
        tuple: Augmented image and label.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    return image, label
