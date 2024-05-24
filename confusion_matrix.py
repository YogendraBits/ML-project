import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Define the precision, recall, f1-score, and support values
precision = [0.91, 0.85]
recall = [0.91, 0.84]
f1_score = [0.91, 0.85]
support = [18788, 11060]

# Define the class labels
classes = ['Class 0', 'Class 1']

# Generate the confusion matrix
cm = np.array([[int(precision[0]*support[0]), int((1-precision[0])*support[0])],
               [int((1-recall[1])*support[1]), int(recall[1]*support[1])]])

# Plot confusion matrix with labels
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Predicted 0 (TN, FP)', 'Predicted 1 (FN, TP)'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
