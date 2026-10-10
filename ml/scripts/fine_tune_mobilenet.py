import tensorflow as tf
from tensorflow.keras import layers

# Paths
train_dir = r"C:\Users\Acer\Desktop\CropSage\ml\dataset_split\train"
validation_dir = r"C:\Users\Acer\Desktop\CropSage\ml\dataset_split\validation"
model_path = r"C:\Users\Acer\Desktop\CropSage\ml\models\mobilenetv2.keras"

# Settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
validation_ds = validation_ds.prefetch(AUTOTUNE)

# Load our already-trained MobileNetV2 model
model = tf.keras.models.load_model(model_path)

# Find the MobileNetV2 base model
base_model = None

for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        base_model = layer
        break

if base_model is None:
    raise ValueError("MobileNetV2 base model was not found.")

# Unfreeze the base model
base_model.trainable = True

# Freeze most layers and fine-tune only the last 20 layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Keep BatchNormalization layers frozen for stable fine-tuning
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

# Use a very small learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Fine-tune
history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=10
)

# Save as a NEW model
fine_tuned_path = r"C:\Users\Acer\Desktop\CropSage\ml\models\mobilenetv2_finetuned.keras"

model.save(fine_tuned_path)

print("\nFine-tuning completed!")
print("Fine-tuned model saved successfully.")