import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Paths
test_dir = r"C:\Users\Acer\Desktop\CropSage\ml\dataset_split\test"
model_path = r"C:\Users\Acer\Desktop\CropSage\ml\models\mobilenetv2_finetuned.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names

print("Classes:", class_names)
print("Number of test images:", len(test_ds.file_paths))

# Load fine-tuned model
model = tf.keras.models.load_model(model_path)

# Evaluate model
loss, accuracy = model.evaluate(test_ds)

print("\nTest Loss:", round(loss, 4))
print("Test Accuracy:", round(accuracy, 4))
print("Test Accuracy (%):", round(accuracy * 100, 2), "%")

# Generate predictions
y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

# Classification report
print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))