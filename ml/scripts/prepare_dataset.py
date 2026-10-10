import os
import shutil

source = r"C:\Users\Acer\Desktop\PlantVillage-Dataset-master\raw\color"
destination = r"C:\Users\Acer\Desktop\CropSage\ml\dataset"

classes = {
    "Tomato___healthy": "Tomato_Healthy",
    "Tomato___Early_blight": "Tomato_Early_Blight",
    "Tomato___Late_blight": "Tomato_Late_Blight",
    "Potato___healthy": "Potato_Healthy",
    "Potato___Early_blight": "Potato_Early_Blight",
    "Potato___Late_blight": "Potato_Late_Blight",
    "Pepper,_bell___healthy": "Pepper_Healthy",
    "Pepper,_bell___Bacterial_spot": "Pepper_Bacterial_Spot"
}

for source_class, destination_class in classes.items():
    source_folder = os.path.join(source, source_class)
    destination_folder = os.path.join(destination, destination_class)

    os.makedirs(destination_folder, exist_ok=True)

    count = 0

    for file in os.listdir(source_folder):
        source_file = os.path.join(source_folder, file)

        if os.path.isfile(source_file):
            shutil.copy2(source_file, destination_folder)
            count += 1

    print(f"{destination_class}: {count} images copied")

print("\nDataset preparation completed.")