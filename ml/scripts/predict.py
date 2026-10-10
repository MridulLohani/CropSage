import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

# Model
MODEL_PATH = r"C:\Users\Acer\Desktop\CropSage\ml\models\mobilenetv2_finetuned.keras"

# Image settings
IMG_SIZE = (224, 224)

# Classes used during training
CLASS_NAMES = [
    "Pepper_Bacterial_Spot",
    "Pepper_Healthy",
    "Potato_Early_Blight",
    "Potato_Healthy",
    "Potato_Late_Blight",
    "Tomato_Early_Blight",
    "Tomato_Healthy",
    "Tomato_Late_Blight"
]

# Load model
model = tf.keras.models.load_model(MODEL_PATH)


def predict_image(image_path):

    # Load and resize image
    image = load_img(image_path, target_size=IMG_SIZE)

    # Convert image to array
    image_array = img_to_array(image)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    predictions = model.predict(image_array, verbose=0)

    # Get top 3 predictions
    top_indices = np.argsort(predictions[0])[-3:][::-1]

    print("\nTop Predictions:")

    for index in top_indices:
        class_name = CLASS_NAMES[index]
        confidence = float(predictions[0][index]) * 100

        print(f"{class_name}: {confidence:.2f}%")


# Ask user for image path
image_path = input("\nEnter the path of the leaf image: ")

predict_image(image_path)