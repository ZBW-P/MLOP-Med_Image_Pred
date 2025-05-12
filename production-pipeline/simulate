import os
import random
import requests
import time
from torchvision import datasets, transforms
from PIL import Image
import io
import shutil
from datetime import datetime

# Dataset directory
final_eval_dir = "/mnt/object/final_eval"

# Load dataset
dataset = datasets.ImageFolder(root=final_eval_dir)
image_paths = [sample[0] for sample in dataset.samples]
random.shuffle(image_paths)

# Inference server URL
FASTAPI_URL = "http://129.114.27.23:8265/predict/"

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])

# Production data output directory
production_dir = "/mnt/data/production_data/unlabeled"
os.makedirs(production_dir, exist_ok=True)

# Limit to the first 50 images
max_images = 100
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
        result = response.json()
        pred_class = result['pred_class']

        # Log output
        print(f"[{idx}/{max_images}] Sent {os.path.basename(image_path)}, predicted: {pred_class}", flush=True)

        # Save a copy of the image into the production_data/unlabeled/ folder
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        # New filename format: timestamp_originalname_pred-<pred_class>_unlabeled.jpg
        new_filename = f"{timestamp}_{os.path.basename(image_path).split('.')[0]}_pred-{pred_class}_unlabeled.jpg"
        dest_path = os.path.join(production_dir, new_filename)

        # Copy the original image to the new location
        shutil.copy2(image_path, dest_path)

    except Exception as e:
        print(f"[{idx}/{max_images}] Error processing {os.path.basename(image_path)}: {e}", flush=True)

    if idx < max_images:
        time.sleep(30)

# ==== Added for rclone upload ====
print("\n🔁 Uploading production data to MinIO bucket 'production'...")

try:
    upload_command = "rclone copy /mnt/data/production_data/unlabeled minio:production --progress --transfers=8 --checkers=8 --fast-list"
    exit_code = os.system(upload_command)
    if exit_code == 0:
        print("✅ Upload completed successfully.")
    else:
        print(f"⚠️ Upload failed with exit code {exit_code}. Please check rclone config.")
except Exception as e:
    print(f"❌ Exception during upload: {e}")
# ==== End of modification ====
