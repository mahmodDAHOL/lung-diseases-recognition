
import matplotlib.pyplot as plt
import os
import numpy as np
import itertools

def plotAccLoss(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(r'/content/drive/MyDrive/covid19_tf - Copy2/images/ACCLearningcurve.PNG')
    plt.close()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(r'/content/drive/MyDrive/covid19_tf - Copy2/images/LOSSLearningcurve.PNG')
    plt.close()
    
    
    
def plotConfusionMatrix(cm, classes,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)


    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    savePathFig = r'/content/drive/MyDrive/covid19_tf - Copy2/images'
    saveNameFig = os.path.join(savePathFig, title)
    plt.savefig(f'{saveNameFig}.jpg', bbox_inches='tight', dpi=300)
    plt.close()
    