import os

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Preprocessing used for inference using pretrained layers for feature extraction


def preprocess_single_image(img: str, width: int, height: int):
    
    # Read image
    img = cv2.imread(img)

    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to the required size
    img = cv2.resize(img, (width, height))

    # Convert image to array
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    return img


def preprocess_images(dataset_path: str, width: int, height: int, num_classes: int):
    
    # Initialize features and labels arrays
    features = []
    labels = []

    # Initialize an identity matrix to one-hot encode data
    identity_matrix = list(np.eye(num_classes, dtype=np.int32))

    # Initialize a list of class names
    class_names = []

    # Check if folders exist
    for folder in os.listdir(dataset_path):
        assert os.path.isdir(os.path.join(dataset_path, folder))
        class_names.append(folder)

    # For every folder in the dataset path
    for folder in os.listdir(dataset_path):
        file_progress_bar = tqdm(os.listdir(
            os.path.join(dataset_path, folder)))

        # For every image in that folder
        for filename in file_progress_bar:

            # If file is image
            if filename.lower().endswith("png") or filename.lower().endswith("jpg") or filename.lower().endswith("jpeg"):

                # Read the grayscale image
                gray_img = cv2.imread(os.path.join(dataset_path, folder, filename))

                # Convert that image to RGB
                color_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2RGB)

                # Resize to the required size
                img = cv2.resize(color_img, (width, height))

                # Convert image to array
                img = tf.keras.preprocessing.image.img_to_array(img)
                img = np.expand_dims(img, axis=0)

                # Add the image array to the features list
                features.append(img)

                # Add the appropriate row from the identity matrix to the labels list
                # E.g.: If the classes are: COVID, NORMAL and PNEUMONIA,
                # the labels will be [1, 0, 0], [0, 1, 0] and [0, 0, 1].
                labels.append(identity_matrix[class_names.index(folder)])

                file_progress_bar.set_description(f"Loading images from directory {folder}")

    # Convert the features list to a vertical stack array.
    # This gets rid of any array of length 1.
    # E.g.: Let (num_images, 1, width, height, channels) be the shape of the features array before converting to a stack array
    # After conversion, the features array will have the shape of (num_images, width, height, channels)
    features = np.vstack(features)

    # Convert the labels list to an array
    labels = np.array(labels)

    return class_names, features, labels
