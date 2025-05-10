# Streamlit app to read object storage directly using Swift API (token-based access)

import streamlit as st
import pandas as pd
from collections import defaultdict
from swiftclient.client import Connection
import os
import random
from PIL import Image
import tempfile

st.set_page_config(page_title="Medical Image Dataset Dashboard", layout="wide")
st.title("📦 Medical Image Dashboard (via Swift API)")

# ==== Auth Setup with Token ====
token = os.environ.get("OS_TOKEN")
storage_url = os.environ.get("STORAGE_URL")

if not token or not storage_url:
    st.error("❌ Missing OS_TOKEN or STORAGE_URL environment variables. Please check your docker-compose configuration.")
    st.stop()

swift_conn = Connection(preauthurl=storage_url,
                        preauthtoken=token,
                        retries=5)

# ==== Configuration ====
container_name = "object-persist-project42-1"
subsets = ['train', 'val', 'test', 'final_eval']
image_exts = ('.jpg', '.jpeg', '.png')

# ==== Load Object List ====
st.info("Fetching object list from container...")
_, objects = swift_conn.get_container(container_name, full_listing=True)

# ==== Organize Object Info ====
data_info = defaultdict(lambda: defaultdict(lambda: {"count": 0, "paths": []}))

for obj in objects:
    parts = obj['name'].split('/')
    if len(parts) >= 3:
        subset, cls, filename = parts[0], parts[1], parts[2]
        if subset in subsets and filename.lower().endswith(image_exts):
            data_info[subset][cls]["count"] += 1
            data_info[subset][cls]["paths"].append(obj['name'])

# ==== Overview ====
st.header("Overview")
total = 0
for subset in subsets:
    subset_total = sum(data_info[subset][cls]['count'] for cls in data_info[subset])
    st.subheader(f"{subset.capitalize()} - {subset_total} images")
    total += subset_total
st.write(f"✅ Total images: **{total}**")

# ==== Detailed Section ====
for subset in subsets:
    if data_info[subset]:
        st.header(f"📂 {subset.capitalize()} Set")
        for cls in data_info[subset]:
            cls_info = data_info[subset][cls]
            st.subheader(f"📁 {cls} - {cls_info['count']} images")

            if cls_info['paths']:
                samples = random.sample(cls_info['paths'], min(3, len(cls_info['paths'])))
                cols = st.columns(3)
                for i, swift_path in enumerate(samples):
                    headers, content = swift_conn.get_object(container_name, swift_path)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                        tmp_file.write(content)
                        tmp_file.flush()
                        with cols[i]:
                            st.image(tmp_file.name, caption=os.path.basename(swift_path), use_container_width=True)
