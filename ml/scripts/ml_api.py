from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Model path
MODEL_PATH = r"C:\Users\Acer\Desktop\CropSage\ml\models\mobilenetv2_finetuned.keras"

# Image size used during training
IMG_SIZE = (224, 224)

# Class names
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

# Load trained model
print("Loading CropSage model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# --------------------------------
# Health Check
# --------------------------------

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ML API is running"
    })


# --------------------------------
# Disease Prediction
# --------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    # Check whether image was uploaded
    if "image" not in request.files:

        return jsonify({
            "error": "No image uploaded"
        }), 400

    image_file = request.files["image"]

    try:

        # Read uploaded image into memory
        image_bytes = image_file.read()

        # Convert bytes into an object supported by TensorFlow
        image = load_img(
            BytesIO(image_bytes),
            target_size=IMG_SIZE
        )

        # Convert image to NumPy array
        image_array = img_to_array(image)

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # Make prediction
        predictions = model.predict(
            image_array,
            verbose=0
        )

        # Get top 3 predictions
        top_indices = np.argsort(
            predictions[0]
        )[-3:][::-1]

        results = []

        for index in top_indices:

            class_name = CLASS_NAMES[index]

            confidence = float(
                predictions[0][index]
            ) * 100

            results.append({
                "class": class_name,
                "confidence": round(
                    confidence,
                    2
                )
            })

        # Highest-confidence prediction
        predicted_index = top_indices[0]

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        predicted_confidence = float(
            predictions[0][predicted_index]
        ) * 100

        # Return prediction
        return jsonify({

            "prediction": predicted_class,

            "confidence": round(
                predicted_confidence,
                2
            ),

            "top_predictions": results

        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# --------------------------------
# Start API
# --------------------------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True
    )