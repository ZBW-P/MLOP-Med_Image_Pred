import os
import shutil
import random
import yaml
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
import requests
import zipfile
import tarfile

# ========== 加载 YAML ==========
with open("datasets_config.yaml", "r") as f:
    config = yaml.safe_load(f)

datasets_info = config["datasets_info"]
categories = config["categories"]

# ========== 配置 ==========
TARGET_ROOT = Path("merged_dataset")
DOWNLOAD_ROOT = Path("downloads")
RANDOM_SEED = 42
SPLIT_RATIO = {"train": 0.6, "test": 0.1, "val": 0.2, "final_eval": 0.1}
IMG_EXTS = {".jpg", ".jpeg", ".png"}
random.seed(RANDOM_SEED)

# ========== 下载 & 解压 ==========
for dataset, info in datasets_info.items():
    url = info["url"]
    archive_type = info["archive_type"]
    extract_to = DOWNLOAD_ROOT / info["extract_to"]
    archive_path = DOWNLOAD_ROOT / f"{dataset}.{archive_type.replace('.', '')}"

    extract_to.parent.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        print(f"⬇️ Downloading {dataset}...")
        with requests.get(url, stream=True) as r:
            with open(archive_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    else:
        print(f"✅ {dataset} already downloaded.")

    if not extract_to.exists():
        print(f"📦 Extracting {dataset}...")
        if archive_type == "zip":
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(DOWNLOAD_ROOT)
        elif archive_type == "tar.gz":
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(DOWNLOAD_ROOT)
    else:
        print(f"✅ {dataset} already extracted.")

# ========== 收集文件 ==========
all_data = defaultdict(list)
oct_person_groups = defaultdict(lambda: defaultdict(list))

for category, cat_info in categories.items():
    is_oct = cat_info.get("oct", False)
    for source in cat_info["sources"]:
        dataset = source["dataset"]
        paths = source["paths"]
        base_dir = DOWNLOAD_ROOT / datasets_info[dataset]["extract_to"]

        for rel_path in paths:
            full_path = base_dir / Path(rel_path)
            if not full_path.exists():
                print(f"⚠️ Path not found: {full_path}")
                continue

            for root, _, files in os.walk(full_path):
                for file in files:
                    if Path(file).suffix.lower() not in IMG_EXTS:
                        continue
                    src = Path(root) / file
                    renamed = f"{dataset}-{category}-{file}"

                    if is_oct:
                        parts = Path(file).stem.split("-")
                        if len(parts) >= 3:
                            person_id = parts[1]
                            oct_person_groups[category][person_id].append((src, renamed))
                    else:
                        all_data[category].append((src, renamed))

# ========== 划分 OCT 图像 ==========
for category, persons in oct_person_groups.items():
    person_items = list(persons.items())
    random.shuffle(person_items)
    total = len(person_items)

    n_train = int(total * SPLIT_RATIO["train"])
    n_test = int(total * SPLIT_RATIO["test"])
    n_val = int(total * SPLIT_RATIO["val"])

    split_dict = {
        "train": person_items[:n_train],
        "test": person_items[n_train:n_train + n_test],
        "val": person_items[n_train + n_test:n_train + n_test + n_val],
        "final_eval": person_items[n_train + n_test + n_val:]
    }

    for split, groups in split_dict.items():
        for _, files in groups:
            for src, new_name in files:
                dest = TARGET_ROOT / split / category / new_name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)

# ========== 划分非 OCT 图像 ==========
for category, files in all_data.items():
    random.shuffle(files)
    total_len = len(files)
    n_train = int(total_len * SPLIT_RATIO["train"])
    n_test = int(total_len * SPLIT_RATIO["test"])
    n_val = int(total_len * SPLIT_RATIO["val"])

    splits = {
        "train": files[:n_train],
        "test": files[n_train:n_train + n_test],
        "val": files[n_train + n_test:n_train + n_test + n_val],
        "final_eval": files[n_train + n_test + n_val:]
    }

    for split, items in splits.items():
        for src_path, new_name in items:
            dest = TARGET_ROOT / split / category / new_name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_path, dest)

print("✅ 数据集处理完成，已保存至:", TARGET_ROOT)
