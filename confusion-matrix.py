import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# true labels and predicted labels
y_true = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7,]
y_pred = [0, 1, 2, 3, 4, 5, 6, 7,0, 1, 2, 3, 4,4, 4, 7,0, 1, 2, 2, 4, 5, 5, 7,]

# classification report
target_names = [
    "Melanocytic nevi",
    "Melanoma",
    "Benign keratosis-like lesions ",
    "Basal cell carcinoma",
    "Actinic keratoses",
    "Vascular lesions",
    "Dermatofibroma",
    "Normal Skin",
]
# create confusion matrix
cm = confusion_matrix(y_true, y_pred)

# normalize the confusion matrix
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# plot the confusion matrix as an image
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=target_names, yticklabels=target_names,
       xlabel='Predicted label',
       ylabel='True label')
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
# add colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Normalized confusion matrix', rotation=-90, va="bottom")
# Loop over data dimensions and create text annotations.
fmt = '.2f'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()
fig.savefig('confusion_matrix.png')
# plt.imsave('confusion_matrix.png')