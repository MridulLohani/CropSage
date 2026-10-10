import tensorflow as tf
from tensorflow.keras import layers, models
import os

train_dir = r"C:\Users\Acer\Desktop\CropSage\ml\dataset_split\train"
validation_dir = r"C:\Users\Acer\Desktop\CropSage\ml\dataset_split\validation"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_data.class_names

print("Classes:", class_names)

normalization = layers.Rescaling(1.0 / 255)

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    normalization,

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=15
)

os.makedirs(
    r"C:\Users\Acer\Desktop\CropSage\ml\models",
    exist_ok=True
)

model.save(
    r"C:\Users\Acer\Desktop\CropSage\ml\models\baseline_cnn.keras"
)

print("Baseline CNN training completed.")
print("Model saved as baseline_cnn.keras")