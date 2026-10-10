# CropSage
## Intelligent Crop Disease Detection, Advisory and Monitoring System

CropSage is an AI-based project designed to identify crop diseases from leaf images and provide useful crop-health information.

The system uses deep learning to classify leaf images into selected healthy and diseased categories. It also includes a backend for user authentication, prediction records, and disease advisory information.

## Technologies Used

- Python
- TensorFlow and Keras
- MobileNetV2
- Flask
- Node.js
- Express.js
- MongoDB Atlas
- React (frontend development planned)

## Current Features

- Crop leaf image classification
- CNN-based model training
- MobileNetV2 transfer learning and fine-tuning
- Flask API for model predictions
- User registration and login
- JWT-based authentication
- Prediction history API development
- Disease advisory API development

## Supported Categories

The current model supports eight categories:

1. Tomato Healthy
2. Tomato Early Blight
3. Tomato Late Blight
4. Potato Healthy
5. Potato Early Blight
6. Potato Late Blight
7. Pepper Healthy
8. Pepper Bacterial Spot

## Model Evaluation

The fine-tuned MobileNetV2 model achieved 96.73% accuracy on the selected test dataset.

This result represents evaluation on the project dataset and does not guarantee the same accuracy under real-world farming conditions.

## Project Status

CropSage is currently under development.

The machine-learning pipeline and initial backend components have been developed. Frontend development, complete API integration, further testing, and deployment remain ongoing tasks.

## Team Members

- Mridul Lohani
- Harshita Kumari
- Muskan Sharma