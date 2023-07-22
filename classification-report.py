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
target_names = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df", "ns"]
report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
print(report)

matrix = confusion_matrix(y_true, y_pred)
# confusion matrix
print(matrix)

# extract precision, recall, and F1-score for each class
precision = [report[class_name]['precision'] for class_name in target_names]
recall = [report[class_name]['recall'] for class_name in target_names]
f1_score = [report[class_name]['f1-score'] for class_name in target_names]

# create a bar chart of precision, recall, and F1-score for each class
fig, ax = plt.subplots()
ax.bar(target_names, precision, label='Precision')
ax.bar(target_names, recall, label='Recall')
ax.bar(target_names, f1_score, label='F1-score')
ax.set_xlabel('Class')
ax.set_ylabel('Score')
ax.set_ylim([0, 1])
ax.legend()

# save the bar chart as an image to a file
plt.savefig('classification_report.png')


