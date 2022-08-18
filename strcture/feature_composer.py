import os

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tools.extraction_and_metrics import compute_confusion_matrix
from tools.kfold import KFold_cross_validation_split
from tools.plotCurves import plotAccLoss, plotConfusionMatrix
from tools.preprocessing import preprocess_images, preprocess_single_image

from .network import Net


# Feature composer training
def train_feature_composer(composed_dataset_path: str, epochs: int,
                           batch_size: int, num_classes: int, folds: int,
                           lr: float, model_dir: str, class_weight: dict):

    # Preprocess images, returning the classes, features and labels
    class_names, x, y = preprocess_images(dataset_path=composed_dataset_path, width=224,
                                          height=224, num_classes=num_classes)

    # Split data
    X_train, X_test, Y_train, Y_test = KFold_cross_validation_split(features=x, labels=y, n_splits=folds)

    # Normalize
    X_train /= 255
    X_test /= 255

    # Instantiate model
    net = Net(pretrained_model=VGG16(input_shape=(224, 224, 3)),
        num_classes=num_classes,lr=lr, mode="feature_composer", labels=class_names, model_dir=model_dir)

    # Train model
    hist = net.fit(x_train=X_train, y_train=Y_train ,x_test=X_test, y_test=Y_test,
            epochs=epochs, batch_size=batch_size, class_weight=class_weight)
    plotAccLoss(hist)
    print("LOOS and ACCURACY curves are saved")


    # Confusion matrix
    cmat = compute_confusion_matrix(y_true=Y_test, y_pred=net.infer(X_test), mode="feature_composer",
                             num_classes=num_classes // 2)
    classes = []
    for i in class_names:
        if '1' in i:
            classes.append(str(i.split('_')[0]))
    plotConfusionMatrix(cmat, classes, title='confusion of composer')

# Inference
def infer(model_details_dir: str, model_dir: str, model_name: str, input_image: str):
   

    with open(f"{os.path.join(model_details_dir, f'{model_name}.txt')}", "r") as f:
        details = f.read()
    
    num_classes = int(details.split("-|-")[-1])
    labels = details.split("-|-")[:-1]
    
    # Instantiate model
    net = Net(pretrained_model=VGG16(input_shape=(224, 224, 3)),
        num_classes=num_classes,mode="feature_composer", model_dir=model_dir, labels=labels)

    # Load model
    tf.keras.models.load_model(filepath=os.path.join(model_dir, model_name))

    # Check if inputed file is an image
    assert input_image.lower().endswith("png") or input_image.lower().endswith("jpg") or input_image.lower().endswith("jpeg")

    # Preprocess
    img = preprocess_single_image(img = input_image, width=224, height=224)

    # Prediction
    return net.infer(img, use_labels=True)
