import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Define absolute paths
train_dir = "/data/xray_dataset/images/train"
output_base_dir = "/data/final_datasets"
csv_path = "/data/xray_dataset/train.csv"

# Create output directory if it does not exist
os.makedirs(output_base_dir, exist_ok=True)

# Load CSV metadata
df = pd.read_csv(csv_path)

# Build a dictionary mapping image_id to class_name
id_to_class = dict(zip(df['image_id'], df['class_name']))

# Build a list of all samples as (npy_path, class_name)
samples = []
for filename in os.listdir(train_dir):
    if filename.endswith('.npy'):
        image_id = filename.replace('.npy', '')
        npy_path = os.path.join(train_dir, filename)
        class_name = id_to_class.get(image_id, 'Unknown')
        samples.append((npy_path, class_name))

# Create DataFrame for easy manipulation
df_samples = pd.DataFrame(samples, columns=['path', 'class_name'])

# Perform stratified split: 70% train, 30% temp
train_df, temp_df = train_test_split(
    df_samples,
    test_size=0.3,
    stratify=df_samples['class_name'],
    random_state=42
)

# Split temp into 10% val, 20% test (to maintain 7:1:2 ratio)
val_df, test_df = train_test_split(
    temp_df,
    test_size=2/3,
    stratify=temp_df['class_name'],
    random_state=42
)

# Prepare dictionary for splits
dataset_splits = {
    'train': train_df,
    'val': val_df,
    'test': test_df
}

# Process and save each image
for split_name, split_df in dataset_splits.items():
    print(f"Processing {split_name} set with {len(split_df)} samples...")
    for _, row in split_df.iterrows():
        npy_path = row['path']
        class_name = row['class_name']
        image_id = os.path.basename(npy_path).replace('.npy', '')

        # Load the numpy array and convert to PIL Image
        img_array = np.load(npy_path)
        img = Image.fromarray(img_array).convert('L')

        # Build output path
        class_dir = os.path.join(output_base_dir, split_name, class_name)
        os.makedirs(class_dir, exist_ok=True)
        png_path = os.path.join(class_dir, f"{image_id}.png")

        # Save as PNG
        img.save(png_path)

print("Data transformation and split completed successfully.")
