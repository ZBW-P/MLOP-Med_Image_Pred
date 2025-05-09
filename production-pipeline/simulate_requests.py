import os
import random
import requests
import time
from torchvision import datasets, transforms
from PIL import Image
import io

# Specify the dataset directory
final_eval_dir = "/mnt/object/final_eval"

# Load the dataset using ImageFolder
dataset = datasets.ImageFolder(root=final_eval_dir)
image_paths = [sample[0] for sample in dataset.samples]
random.shuffle(image_paths)

# Set the inference server URL
FASTAPI_URL = "http://129.114.27.23:8265/predict/"

# Define the preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])

# Limit to the first 50 images
max_images = 50
selected_images = image_paths[:max_images]

# Loop to send requests every 5 seconds
for idx, image_path in enumerate(selected_images, 1):
    try:
        # Open the original image
        img = Image.open(image_path).convert("RGB")

        # Apply preprocessing
        tensor = preprocess(img)

        # Reverse normalization and convert back to PIL Image for sending
        unnormalize = transforms.Normalize(mean=[-0.5 / 0.5], std=[1 / 0.5])
        img_tensor = unnormalize(tensor)
        img_pil = transforms.ToPILImage()(img_tensor)

        # Save the processed image to an in-memory buffer
        buf = io.BytesIO()
        img_pil.save(buf, format='JPEG')
        buf.seek(0)

        # Send the image to the inference endpoint
        files = {'file': (os.path.basename(image_path), buf, 'image/jpeg')}
        response = requests.post(FASTAPI_URL, files=files)
        print(f"[{idx}/{max_images}] Sent {image_path}, response: {response.json()}")
    except Exception as e:
        print(f"[{idx}/{max_images}] Error processing {image_path}: {e}")

    if idx < max_images:
        time.sleep(5)
