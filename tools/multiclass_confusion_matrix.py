import numpy as np


def multiclass_confusion_matrix(cmat: np.ndarray, num_classes: int):
    
    acc = np.zeros(num_classes)
    sn = np.zeros(num_classes)
    sp = np.zeros(num_classes)

    for i in range(num_classes):
        TP, TN, FP, FN = 0, 0, 0, 0

        TP += cmat[i][i]

        for j in range(0, num_classes):
            if j != i:
                for k in range(0, num_classes):
                    if k != i:
                        TN += cmat[j][k]

        for j in range(0, num_classes):
            if j != i:
                FP += cmat[j][i]
                FN += cmat[i][j]

        acc[i] = (TP + TN) / (TP + TN + FP + FN)
        sn[i] = TP / (TP + FN)
        sp[i] = TN / (TN + FP)

    return np.mean(acc), np.mean(sn), np.mean(sp)
