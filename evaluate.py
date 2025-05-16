import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tensorflow.keras.losses import CategoricalCrossentropy
import seaborn as sns

val_test_datagen = ImageDataGenerator(rescale=1.0/255)

test_generator = val_test_datagen.flow_from_directory(
    config.TEST_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',  # Multi-class classification
    shuffle=False
)

model = tf.keras.models.load_model('cnn_leukemia.h5')
print("Model loaded from cnn_leukemia.h5")

# Evaluate model
test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print(f"Test Accuracy: {test_acc:.2f}")

# Predict probabilities
y_pred_prob = model.predict(test_generator)

# Convert probabilities to class predictions
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = test_generator.classes

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))

# Compute Precision, Recall, and F1-score
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

# Compute ROC Curve and AUC for each class
fpr = {}
tpr = {}
roc_auc = {}
num_classes = len(test_generator.class_indices)
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve((y_true == i).astype(int), y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

macro_auc = roc_auc_score(y_true, y_pred_prob, multi_class='ovr', average='macro')
weighted_auc = roc_auc_score(y_true, y_pred_prob, multi_class='ovr', average='weighted')
print(f"Macro AUC: {macro_auc:.4f}")
print(f"Weighted AUC: {weighted_auc:.4f}")

# Compute Categorical Crossentropy Loss
cce = CategoricalCrossentropy()
# Convert true labels to one-hot encoding
y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=num_classes)
cat_crossentropy_loss = cce(y_true_onehot, y_pred_prob).numpy()
print(f"Categorical Crossentropy Loss: {cat_crossentropy_loss:.4f}")

# Plot ROC Curves
plt.figure(figsize=(10, 7))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()