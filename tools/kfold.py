import numpy as np
from sklearn.model_selection import KFold

def KFold_cross_validation_split(features: np.ndarray, labels: np.ndarray, n_splits: int):
    
    kfold = KFold(n_splits=n_splits, shuffle=True)
    for train_idx, test_idx in kfold.split(features):
        x_train, x_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

    return x_train, x_test, y_train, y_test
