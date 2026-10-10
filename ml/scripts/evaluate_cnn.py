import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

model_path = r"C:\Users\Acer\Desktop\CropSage\ml\models\baseline_cnn.keras"
test_dir = r"C:\Users\Acer\Desktop\CropSage\ml\dataset_split\test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_data.class_names

model = tf.keras.models.load_model(model_path)

loss, accuracy = model.evaluate(test_data)

print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

y_true = np.concatenate([y.numpy() for x, y in test_data])

predictions = model.predict(test_data)
y_pred = np.argmax(predictions, axis=1)

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))