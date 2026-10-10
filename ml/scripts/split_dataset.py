import os
import shutil
import random

source_dir = r"C:\Users\Acer\Desktop\CropSage\ml\dataset"
output_dir = r"C:\Users\Acer\Desktop\CropSage\ml\dataset_split"

classes = [
    "Tomato_Healthy",
    "Tomato_Early_Blight",
    "Tomato_Late_Blight",
    "Potato_Healthy",
    "Potato_Early_Blight",
    "Potato_Late_Blight",
    "Pepper_Healthy",
    "Pepper_Bacterial_Spot"
]

train_ratio = 0.70
validation_ratio = 0.15

random.seed(42)

for class_name in classes:
    class_path = os.path.join(source_dir, class_name)
    images = os.listdir(class_path)

    random.shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    validation_end = train_end + int(total * validation_ratio)

    train_images = images[:train_end]
    validation_images = images[train_end:validation_end]
    test_images = images[validation_end:]

    for split_name, split_images in [
        ("train", train_images),
        ("validation", validation_images),
        ("test", test_images)
    ]:
        destination = os.path.join(output_dir, split_name, class_name)
        os.makedirs(destination, exist_ok=True)

        for image in split_images:
            shutil.copy2(
                os.path.join(class_path, image),
                os.path.join(destination, image)
            )

    print(
        f"{class_name}: "
        f"Train={len(train_images)}, "
        f"Validation={len(validation_images)}, "
        f"Test={len(test_images)}"
    )

print("\nDataset splitting completed.")