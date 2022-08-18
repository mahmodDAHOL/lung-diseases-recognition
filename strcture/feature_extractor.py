import os

import numpy as np
from tensorflow.keras.applications import VGG16
from tools.extraction_and_metrics import (compute_confusion_matrix,
                                          extract_features)
from tools.kfold import KFold_cross_validation_split
from tools.plotCurves import plotConfusionMatrix
from tools.preprocessing import preprocess_images

from .network import Net


# Feature extractor training
def train_feature_extractor(initial_dataset_path: str, extracted_features_path: str,
                            epochs: int, batch_size: int, num_classes: int, folds: int,
                            lr: float, model_dir: str):
   
    class_names, x, y = preprocess_images(dataset_path=initial_dataset_path, width=224,
                                          height=224, num_classes=num_classes)

    X_train, X_test, Y_train, Y_test = KFold_cross_validation_split(features=x, labels=y, n_splits=folds)
    X_train /= 255
    X_test /= 255
    net = Net(pretrained_model=VGG16(input_shape=(224, 224, 3), include_top=True),
        num_classes=num_classes, lr=lr, mode="feature_extractor", labels=class_names, model_dir=model_dir)

    net.fit(x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test,
            epochs=epochs, batch_size=batch_size, class_weight=None)

    cmat = compute_confusion_matrix(y_true=Y_test, y_pred=net.infer(X_test, use_labels=False),
                              mode="feature_extractor", num_classes = num_classes)
    plotConfusionMatrix(cmat, class_names, title='confusion of extractor')
    
    for class_name in class_names:
        extracted_features = extract_features(initial_dataset_path=initial_dataset_path,
                                              class_name=class_name, width=224, height=224, net=net)
        np.save(
            file=os.path.join(extracted_features_path, f"{class_name}.npy"), arr=extracted_features)
