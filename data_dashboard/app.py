import streamlit as st
import os
import random
from PIL import Image

# Define the data path
DATA_ROOT = "/mnt/object"

# Function to get dataset info
def get_dataset_info():
    data_info = {}
    subsets = ['train', 'val', 'test', 'final_eval']
    for subset in subsets:
        subset_path = os.path.join(DATA_ROOT, subset)
        if os.path.exists(subset_path):
            classes = os.listdir(subset_path)
            data_info[subset] = {}
            for cls in classes:
                cls_path = os.path.join(subset_path, cls)
                if os.path.isdir(cls_path):
                    num_images = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    data_info[subset][cls] = {
                        "count": num_images,
                        "sample_images": random.sample(
                            [os.path.join(cls_path, f) for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
                            min(3, num_images)
                        ) if num_images > 0 else []
                    }
    return data_info

# Page title
st.title("Medical Image Dataset Dashboard")

# Load dataset info
data_info = get_dataset_info()

# Overview section
st.header("Dataset Overview")

total_images = 0
for subset in data_info:
    subset_total = sum(data_info[subset][cls]['count'] for cls in data_info[subset])
    st.subheader(f"{subset.capitalize()} - {subset_total} images")
    total_images += subset_total

st.write(f"Total number of images across all subsets: {total_images}")

# Detailed information
for subset in data_info:
    st.header(f"{subset.capitalize()} Set")
    for cls in data_info[subset]:
        cls_info = data_info[subset][cls]
        st.subheader(f"{cls} - {cls_info['count']} images")

        cols = st.columns(3)
        for i, img_path in enumerate(cls_info['sample_images']):
            with cols[i]:
                st.image(Image.open(img_path), caption=os.path.basename(img_path), use_column_width=True)
