import os
import shutil
import random
import yaml
from pathlib import Path
from collections import defaultdict
import requests
import zipfile
import tarfile

# ===== Load YAML =====
with open("datasets_config.yaml", "r") as f:
    config = yaml.safe_load(f)

datasets_info = config["datasets_info"]
categories = config["categories"]

# ===== Config =====
TARGET_ROOT = Path("merged_dataset")
DOWNLOAD_ROOT = Path("downloads")
RANDOM_SEED = 42
IMG_EXTS = {".jpg", ".jpeg", ".png"}
random.seed(RANDOM_SEED)

#===== Download & Extract =====
for dataset, info in datasets_info.items():
    url = info["url"]
    archive_type = info["archive_type"]
    extract_to = DOWNLOAD_ROOT / info["extract_to"]
    archive_path = DOWNLOAD_ROOT / f"{dataset}.{archive_type.replace('.', '')}"

    extract_to.parent.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        print(f"Downloading {dataset}...")
        with requests.get(url, stream=True) as r:
            with open(archive_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    else:
        print(f"{dataset} already downloaded.")

    if not extract_to.exists():
        print(f"Extracting {dataset}...")
        if archive_type == "zip":
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(DOWNLOAD_ROOT)
        elif archive_type == "tar.gz":
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(DOWNLOAD_ROOT)
    else:
        print(f"{dataset} already extracted.")

# for dataset, info in datasets_info.items():
#     extract_to = DOWNLOAD_ROOT / info["extract_to"]
#     print(f"Skipping download & extraction for {dataset}. Using existing data at: {extract_to}")

# ===== Collect Files =====
all_data = defaultdict(list)
oct_person_groups = defaultdict(lambda: defaultdict(list))

print("\nCollecting files...")
for category, cat_info in categories.items():
    is_oct = cat_info.get("oct", False)
    for source in cat_info["sources"]:
        dataset = source["dataset"]
        paths = source["paths"]
        base_dir = DOWNLOAD_ROOT / datasets_info[dataset]["extract_to"]

        for rel_path in paths:
            full_path = base_dir / Path(rel_path)
            if not full_path.exists():
                print(f"[Warning] Path not found: {full_path}")
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

# ===== Split OCT Images =====
for category, persons in oct_person_groups.items():
    print(f"\nProcessing OCT category: {category}")
    person_items = list(persons.items())
    random.shuffle(person_items)
    total = len(person_items)
    print(f"Total persons: {total}")

    n_train = int(total * 0.7)
    rest = person_items[n_train:]

    # Split rest into test + val
    n_test = int(len(rest) * 0.33)
    test_val = rest[:]
    test = test_val[:n_test]
    val = test_val[n_test:]

    # From test + val, pick 50% of total as final_eval
    combined_for_eval = test + val
    n_final_eval = int(len(combined_for_eval) * 0.5)  # 50% of test+val
    final_eval = combined_for_eval[:n_final_eval]


    split_dict = {
        "train": person_items[:n_train],
        "test": test,
        "val": val,
        "final_eval": final_eval,
    }

    for split, groups in split_dict.items():
        count = 0
        for _, files in groups:
            for src, new_name in files:
                dest = TARGET_ROOT / split / category / new_name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
                count += 1
        print(f"{split}: {count} images saved.")

# ===== Split Non-OCT Images =====
for category, files in all_data.items():
    print(f"\nProcessing non-OCT category: {category}")
    random.shuffle(files)
    total_len = len(files)
    print(f"Total images: {total_len}")

    n_train = int(total_len * 0.7)
    rest = files[n_train:]

    # Split rest into test + val
    n_test = int(len(rest) * 0.33)
    test_val = rest[:]
    test = test_val[:n_test]
    val = test_val[n_test:]


    # Random sample from test + val
    combined_for_eval = test + val
    random.shuffle(combined_for_eval)
    n_final_eval = int(len(combined_for_eval) * 0.5)  # 取 test+val 的 50%
    final_eval = combined_for_eval[:n_final_eval]


    split_map = {
        "train": files[:n_train],
        "test": test,
        "val": val,
        "final_eval": final_eval,
    }

    for split, items in split_map.items():
        for src_path, new_name in items:
            dest = TARGET_ROOT / split / category / new_name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_path, dest)
        print(f"{split}: {len(items)} images saved.")

print("\nDataset processing completed. Merged dataset saved to:", TARGET_ROOT.resolve())
