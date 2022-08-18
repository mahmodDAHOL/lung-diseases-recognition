import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, Sequential, save_model
from tensorflow.keras.optimizers import SGD


# Callback used for saving
class DeTraC_callback(tf.keras.callbacks.Callback):
    def __init__(self, model: Sequential, num_epochs: int, filepath: str):
        super(DeTraC_callback, self).__init__()
        self.model = model
        self.num_epochs = num_epochs
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0 or self.num_epochs < 10:
            save_model(model=self.model, filepath=self.filepath, save_format='tf')

# The network
class Net(object):
    """
    The DeTraC model.

    params:
            <Sequential> the model
            <int> num_classes
            <float> lr: Learning rate
            <string> mode: The DeTraC model contains 2 modes which are used depending on the case:
                                - feature_extractor: used in the first phase of computation, where the pretrained model is used to extract the main features from the dataset
                                - feature_composer: used in the last phase of computation, where the model is now training on the composed images, using the extracted features and clustering them.
            <string> model_dir
            <list> labels: List of labels
    """

    def __init__(self, pretrained_model: Model, num_classes: int, mode: str,
                 model_dir: str, labels: list = [], lr: float = 0.0):

        self.pretrained_model = pretrained_model
        self.mode = mode
        self.num_classes = num_classes
        self.lr = lr
        self.labels = labels
        self.model_dir = model_dir

        # Check if model directory exists
        assert os.path.exists(self.model_dir)
        
        self.model_details_dir = os.path.join(model_dir, "details")
        
        # Check whether mode is correct
        assert self.mode == "feature_extractor" or self.mode == "feature_composer"

        now = datetime.now()
        now = f'{str(now).split(" ")[0]}_{str(now).split(" ")[1]}'.split(".")[0].replace(':', "-")

        # Initialize custom weights
        self.custom_weights = lambda shape, dtype = None: \
            tf.Variable(lambda: tf.random.normal(shape) * 0.0001)

        # Initialize custom biases
        self.custom_biases = lambda shape, dtype = None: \
            tf.Variable(lambda: tf.random.normal(shape) * 0.0001 + 1)


#         self.input_layer = self.pretrained_model.layers[0]
            
        # Pretrained layers
        self.pretrained_layers = self.pretrained_model.layers[1:-3]
        initializer = tf.keras.initializers.HeNormal()
        # Custom fully conncted layer
        self.classification_layer = Dense(
            units=48,
            activation='relu',
            kernel_initializer=initializer,
        )
        # Custom classification layer
        self.classification_layer = Dense(
            units=self.num_classes,
            activation='softmax',
            kernel_initializer=self.custom_weights,
            bias_initializer=self.custom_biases
        )

        if self.mode == "feature_extractor":
            self.save_name = f"DeTraC_feature_extractor_{now}"
            for layer in self.pretrained_layers:
                layer.trainable = True
                
            self.classification_layer.trainable = True
                
            self.optimizer = SGD(learning_rate=self.lr, momentum=0.9, nesterov=False, decay=1e-3)
            self.scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.9, patience=3)
        else:
            for layer in self.pretrained_layers:
                    layer.trainable = True
                
            self.classification_layer.trainable = True
            
            assert len(labels) == num_classes
            self.save_name = f"DeTraC_feature_composer_{now}"
            self.optimizer = SGD(learning_rate=self.lr, momentum=0.95, nesterov=False, decay=1e-4)
            self.scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.95, patience=5)
            
        self.layers = []
        for layer in self.pretrained_layers:
            self.layers.append(layer)
        self.layers.append(self.classification_layer)
        
        # Instantiate model
        self.model = Sequential(self.layers)
        self.model_path = os.path.join(self.model_dir, self.save_name)
        self.model_details_path = os.path.join(self.model_details_dir, f"{self.save_name}.txt")

        # Compile model
        self.model.compile(optimizer=self.optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    def save_for_inference(self):
        """
        Saves the label names and number of classes for inference.
        """
        with open(self.model_details_path, "w") as f:
            for label in self.labels:
                f.write(f"{label}-|-")
            f.write(str(self.num_classes))
        
    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
            y_test: np.ndarray, epochs: int, batch_size: int, class_weight: dict):

        # If the feature composer is being used
        if self.mode == "feature_composer":
            # Save details (number of classes and labels)
            self.save_for_inference()
    
            # [DATA AUGMENTATION] Instantiate an image data generator
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                featurewise_std_normalization = True,
                horizontal_flip = True
            )
            datagen.fit(x_train)

        # Instantiate the custom DeTraC callback
        custom_callback = DeTraC_callback(model=self.model, num_epochs=epochs, filepath=self.model_path)

        
        if self.mode == "feature_extractor":
            self.model.fit(
                x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
                validation_data=(x_test, y_test), validation_freq=1,
                shuffle=True, verbose=1# , callbacks=[self.scheduler, custom_callback]
                )
        else:
            history = self.model.fit(
                x=datagen.flow(x_train, y_train, batch_size=batch_size),
                epochs=epochs, validation_data=(x_test, y_test), validation_freq=1,
                steps_per_epoch=len(x_train) // batch_size, shuffle=True,
                verbose=1, # callbacks=[self.scheduler, custom_callback],
                class_weight=class_weight)
            return history
    # Inference
    def infer(self, input_data: np.ndarray, use_labels: bool = False):
        """
        Inference function.

        params:
            <array> input_data
            <bool> use_labels: Whether to output nicely, with a description of the labels, or not
        returns:
            <array> prediction
        """

        # Prediction
        output = self.model.predict(input_data)
        if use_labels == True:
            labeled_output = {}
            labels = self.labels
            for label, out in zip(labels, output[0]):
                labeled_output[label] = out
            return labeled_output
        else:
            return output

    def infer_using_pretrained_layers_without_last(self, features: np.ndarray):
        """
        Feature extractor's inference function.

        params:
            <array> features
        returns:
            <array> NxN array representing the features of an image
        """
        # Instantiate a sequential model
        extractor = Sequential()
        
        # Add all the pretrained layers to it
        for layer in self.model.layers[0:]:
            extractor.add(layer)

        # Use the extractor to predict upon the input image
        output = extractor.predict(features)
        return output
