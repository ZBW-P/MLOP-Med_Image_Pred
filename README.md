# Project Title: [Your Project Name]

## Project Overview

Our project develops a **hybrid machine learning system** that integrates a Vision Transformer (ViT) for initial disease classification from chest X-ray images with a Large Language Model (LLM) for interpretability and actionable insights. In current clinical settings, radiologists manually assess chest X-rays—a process that is time-consuming, subjective, and prone to inconsistency. Our system addresses these issues while satisfying the project’s requirements for scale in data, model complexity, and deployment.

## Key Value Propositions

### 1. User Preview and Enhanced Communication
- **Preliminary Results:** Provides users with an immediate preliminary assessments of potential chest abnormalities from their X-ray images.
- **Improved Communication:** Helps patients articulate their symptoms more effectively during consultations, especially while in the waiting room.
- **Fast, Accessible Feedback:** Thanks to the model’s efficient design and quick update mechanism, users receive real-time preliminary assessments. Although the model may trade off some accuracy compared to professional-grade systems, its speed and ease-of-access allow patients to obtain useful insights promptly.

### 2. Reduction of Patient Revisits
- **Initial Prediction:** Offers an early indication of whether findings are mild or severe, which can reduce the need for multiple doctor visits (reducing the cost per visit).
- **Feedback Loop:** Physician-confirmed feedback is incorporated to refine future predictions, ensuring that the system’s recommendations remain safe and reliable.

### 3. Lightweight and Accessible Deployment
- **Balanced Performance:** Our model is designed to be medium-sized, striking a balance between computational efficiency and performance. This enables easy deployment on websites and mobile applications without requiring extensive hardware resources.
- **Multi-Platform Use:** Its lightweight nature means it can be integrated into various devices, ensuring that users have quick access to preliminary predictions.
- **Efficient Model Size & Storage:** Unlike professional models that are large and resource-intensive, our system is optimized for lower storage needs and faster computation. This makes it easier to update and maintain, while still delivering valuable insights.

### 4. Automated Feedback and Model Upgrades
- **Continuous Improvement:** User interactions and professional feedback are automatically managed via our dedicated web platform and server, creating a self-improving system.
- **Model Supervision:** Ongoing monitoring and regular upgrades ensure that the model remains clinically relevant over time.
- **Optimized Update Mechanism:** By leveraging an online server infrastructure for both image storage and model updates, the system maintains a balanced trade-off between model size and accuracy. This allows patients, who might otherwise face repeated doctor visits, to benefit from timely and cost-effective preliminary assessments.

### Disclaimer

**Important Notice for Patients:**  
Our model is a medium-sized model designed for easy deployment on websites and mobile apps. However, its accuracy is not guaranteed. The outputs provided by the system serve as a preliminary preview and should not replace professional medical advice. **Always consult with a professional doctor before relying on these results.**

### Contributors

<!-- Table of contributors and their roles. 
First row: define responsibilities that are shared by the team. 
Then, each row after that is: name of contributor, their role, and in the third column, 
you will link to their contributions. If your project involves multiple repos, you will 
link to their contributions in all repos here. -->

| Name                            | Responsible for | Link to their commits in this repo |
|---------------------------------|-----------------|------------------------------------|
| All team members                |                 |                                    |
| Qin Huai               |Model training and training platforms                 | https://github.com/ZBW-P/MLOP-Med_Image_Pred?tab=readme-ov-file#model-training-and-training-platforms                                  |
| Zhaochen Yang                  |Model Serving, Evaluation and Monitoring                |https://github.com/ZBW-P/MLOP-Med_Image_Pred/blob/main/README.md#model-serving-and-monitoring-platforms                                    |
| Junjie Mai                   |Data Pipeline                 |https://github.com/ZBW-P/MLOP-Med_Image_Pred/blob/main/README.md#data-pipeline                                    |
| Hongshang Fan |Continuous X                 |https://github.com/ZBW-P/MLOP-Med_Image_Pred/blob/main/README.md#Continuous-X                                    |



### System diagram

<!-- Overall digram of system. Doesn't need polish, does need to show all the pieces. 
Must include: all the hardware, all the containers/software platforms, all the models, 
all the data. -->
![Diagram](https://github.com/ZBW-P/MLOP-Med_Image_Pred/blob/main/System.png)

### Summary of outside materials

<!-- In a table, a row for each dataset, foundation model. 
Name of data/model, conditions under which it was created (ideally with links/references), 
conditions under which it may be used. -->

Our overall approach is to leverage the dataset detailed in the table below as the foundation for our ML operations system. Our pipeline integrates persistent storage, device-based processing, and Docker orchestration to streamline training, evaluation, and validation. Specifically, we plan to:

- **Centralize Data Storage:**  
  Store the collected chest X-ray and OCT images in a clear and organized library on persistent storage. The data will be segmented into three zipped folders (`train`, `test`, and `val`), ensuring that all training, evaluation, and testing sets are easily accessible to our system.
  
- **Model training and evaluation:**
  We will train a custom Vision Transformer (ViT) model on chest X-ray images for accurate medical image classification. A LitGPT-derived LLM will be employed to generate diagnostic derivations and actionable recommendations based on the outputs of the ViT model. Our training pipeline will support continuous training, evaluation, and re-training on new or updated data to ensure that both models remain current and adapt to evolving clinical requirements.
    
- **Integration with Our Pipeline and Device Infrastructure:**  
  Use Python scripts to read and preprocess images from the persistent storage. These scripts will run on dedicated devices (e.g., GPU-enabled servers) that are integrated into our Docker-based ML operations system. The images will be resized and formatted as needed based on our model requirements, ensuring that data is properly prepared for ingestion into our training pipeline.


The table below summarizes the datasets we plan to use, including details on how each dataset was created and the conditions under which it may be used.

| Name of Data/Model                              | How it was Created                                                                                                                                                                                                                                                                                                                                                                                                                 | Conditions of Use                                                                                                                                                                                                                                                       |
|-------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Labeled Optical Coherence Tomography (OCT)**              | Dataset of validated OCT described and analyzed in "Deep learning-based classification and referral of treatable human diseases". The OCT Images are split into a training set and a testing set of independent patients. OCT Images are labeled as (disease)-(randomized patient ID)-(image number by this patient) and split into 4 directories: CNV, DME, DRUSEN, and NORMAL. More details can be found on the [Dataset Page](https://data.mendeley.com/datasets/rscbjbr9sj/2)). |The files associated with this dataset are licensed under a Creative Commons Attribution 4.0 International licence. You can share, copy and modify this dataset so long as you give appropriate credit, provide a link to the CC BY license, and indicate if changes were made, but you may not do so in a way that suggests the rights holder has endorsed you or your use of the dataset. Note that further permission may be required for any content within the dataset that is identified as belonging to a third party.                                                                  |
| **COVID-19 Radiography Database**              | Developed by a collaborative team from Qatar University, the University of Dhaka, and partner institutions, this dataset aggregates chest X-ray images for COVID-19, Normal, and Viral Pneumonia cases. The initial release provided 219 COVID-19, 1341 Normal, and 1345 Viral Pneumonia images, with updates increasing the COVID-19 cases to 3616 and including corresponding lung masks. More details can be found on the [Kaggle Dataset Page](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/data). | Released for academic and non-commercial use. Users must cite the original publications (Chowdhury et al., IEEE Access, 2020; Rahman et al., 2020[Publication1](https://ieeexplore.ieee.org/document/9144185),[Publication2](https://www.sciencedirect.com/science/article/pii/S001048252100113X?via%3Dihub) and adhere to the dataset usage guidelines provided in its metadata.                                                                     |
| **Tuberculosis (TB) Chest X-ray Database** | Developed by a collaborative research team from Qatar University, University of Dhaka, Malaysia, and affiliated medical institutions, this database comprises chest X-ray images for TB-positive cases and Normal images. The current release includes 700 publicly accessible TB images, 2800 additional TB images available via a data-sharing agreement through the NIAID TB portal, and 3500 Normal images. The dataset is compiled from multiple sources (including the NLM, Belarus, NIAID TB, and RSNA CXR datasets) and was used in the study “Reliable Tuberculosis Detection using Chest X-ray with Deep Learning, Segmentation and Visualization” published in IEEE Access. For further details, refer to the [Kaggle Dataset Page](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset). | Licensed for academic and non-commercial research use. Users must provide proper citation to the original publication: Tawsifur Rahman, Amith Khandakar, Muhammad A. Kadir, Khandaker R. Islam, Khandaker F. Islam, Zaid B. Mahbub, Mohamed Arselene Ayari, Muhammad E. H. Chowdhury. (2020) “Reliable Tuberculosis Detection using Chest X-ray with Deep Learning, Segmentation and Visualization”. IEEE Access, Vol. 8, pp 191586–191601 (DOI: 10.1109/ACCESS.2020.3031384), and adhere to the dataset usage guidelines. |
| **Vision Transformer (ViT) Demo Model**        | Defined in PyTorch in Prof. Hegde’s `visual_transformers.ipynb` (dl-demos repository). It implements: patch embedding (16×16), positional embeddings, multi-head self-attention, transformer encoder layers, and an MLP head. The notebook provides a full training loop (DataLoader, Adam optimizer, cross-entropy loss, scheduler) for experimentation on standard image datasets. | For academic/course use only, per the deep-learning course guidelines; not for commercial distribution.|

### Summary of infrastructure requirements

<!-- Itemize all your anticipated requirements: What (`m1.medium` VM, `gpu_mi100`), 
how much/when, justification. Include compute, floating IPs, persistent storage. 
The table below shows an example, it is not a recommendation. -->

| Requirement     | How many/when                                     | Justification |
|-----------------|---------------------------------------------------|---------------|
| `m1.xlarge` VMs | 3 for entire project duration, 1 for model training                   |           |
| `gpu_m100`     | 2 GPUs for model training (using DDP/FSDP)   |The ViT model's training on a large dataset with significant parameters benefits from parallel GPU processing (2 GPUs) to efficiently handle computations|
| Floating IPs    | 1 for the entire project duration and 1 for training use | One persistent IP ensures continuous connectivity to services throughout the project, while an additional floating IP offers flexibility for Ray train and model training needs.                |
| Outside memory / storage            | 100 g storage during all project duration                                           | Large update medical image data need to be saved and used to update for server and user interacter.               |

## Unit 2/3: Cloud-Native Infrastructure

**Architecture Diagram**  



### Provisioning Summary

![infrastructure](https://github.com/user-attachments/assets/1b30abd6-74c2-4c73-8b6a-ab322f033f0a)
We provisioned resources on **Chameleon Cloud (KVM@TACC)** using the **OpenStack CLI**, and configured services via the **Jupyter interface** by running [`infrastructure_provision/provision_cli.ipynb`](infrastructure_provision/provision_cli.ipynb).

- **Private Network**: `private-net-project42`
- **Nodes**:
  - `node1-mlops-project42-1` (192.168.1.11 / public: 129.114.27.23)
  - `node2-mlops-project42-1` (192.168.1.12)
  - `node3-mlops-project42-1` (192.168.1.13)
- **Flavor**: `m1.xlarge` (8 vCPUs, 16GB RAM, 40GB disk)
- **Image**: `CC-Ubuntu24.04`
- **Volume attached**: `block-persist-project42-1` on `/dev/vdb`

### Security Group for Services 

| Service       | Port | Container                | Security Group |
|---------------|------|--------------------------|----------------|
| MinIO         | 9000/9001 | `minio`             | `allow-9000`, `allow-9001` |
| MLflow        | 8000 | `mlflow`                 | `allow-8000`   |
| PostgreSQL    | 5432 | `postgres`               | (default access) |
| Prometheus    | 9090 | `prometheus`             | `allow-9090`   |
| Grafana       | 3000 | `grafana`                | `allow-3000`   |
| ViT API       | 8265 | `vit-container`          | `allow-8265`   |
| Streamlit Dashboard | 9002 | `practical_knuth` | `allow-9002`   |
| Jupyter       | 8888 | `jupyter`                |  `Allow HTTP 8888` |

All services were launched using `docker run` or `docker-compose` from within Jupyter on `node1`.


## Unit 8 (Offline): Persistent Storage and Data Pipeline

### Persistent Storage

We use two persistent storage layers in our system:

#### 1. Block Storage (Chameleon KVM@TACC)

Provisioned on **Chameleon Cloud** and mounted on `node1-mlops-project42-1` at `/dev/vdb`.

- **Volume Name**: `block-persist-project42-1`
- **Size**: 30 GB
- **Type**: `ceph-hdd`
- **Attached To**: `node1-mlops-project42-1` (`/dev/vdb`)
- **Used For**:
  - MinIO data (`/mnt/block/minio_data`)
  - PostgreSQL storage (`/mnt/block/postgres_data`)

#### 2. Object Storage (Chameleon CHI@TACC)

Used for storing all dataset splits for model training and evaluation.

- **Bucket**: `object-persist-project42`
- **Size**: 7.70 GB
- **Object Count**: 125,100
- **Structure**:  
  - `train/`  
  - `val/`  
  - `test/`  
  - `final_eval/`

This object store is read-only mounted into the Jupyter container at `/mnt/medical-data` for training and inference.

**Offline Dataset**  
- Datasets used: Chest X-ray, OCT, etc.  
- Scripts: [`data/prepare_data.py`](link)  
- Sample illustration: insert or link representative samples

**Data Pipeline**  
- Source → Preprocessing → Object Storage  
- Splitting strategy: train/val/test/final_eval (mention person ID integrity if applicable)  
- Data cleaning steps (resizing, normalization, etc.)

### Data Dashboard
# Medical Image Dashboard (Swift API via Streamlit)
This is a lightweight Streamlit dashboard for visualizing medical image datasets stored in OpenStack Swift object storage.

![dashboard preview](https://github.com/user-attachments/assets/6619c926-3e7d-4f4a-8c8c-859dba869f26)

## Features
- Connects to a Swift container using token-based authentication
- Shows image counts for train, val, test, and final_eval folders
- Displays sample images (randomly selected) per class
- Runs as a web application on port 9002 using Docker

## How to run it
Use the following Python script in your Chameleon Jupyter environment to authenticate and retrieve the necessary credentials for accessing OpenStack Swift object storage.

```python
from chi import server, context
import chi, os, time, datetime

# Select your project and site
context.choose_project()
context.choose_site(default="CHI@TACC")

# Establish a connection to the OpenStack environment
from chi import clients
conn = clients.connection()

# Print authentication token and object storage endpoint
print("OS_TOKEN =", conn.authorize())
print("STORAGE_URL =", conn.object_store.get_endpoint())
```
Build the Docker Image
```bash
docker build -t swift-dashboard .
```
Run the Dashboard
Replace the variables with your actual token and Swift storage URL:
```bash
docker run -p 9002:9002 \
  -e OS_TOKEN="your_token_here" \
  -e STORAGE_URL="your_storage_url_here" \
  -v $(pwd):/app \
  swift-dashboard
```
Access the Dashboard on http://129.114.27.23:9002

## Unit 4 & 5: Modeling, Training, and Experiment Tracking

### Problem Setup & Model Motivation

This project addresses the challenge of classifying nine distinct lung disease categories (e.g., lung-covid, lung-oct-cnv, lung-oct-drusen, lung-opacity, lung-viral-pneumonia, etc.) from chest X-ray and OCT images. Early and accurate detection of these conditions is critical for patient care, but traditional convolutional neural networks can struggle to capture small, localized lesions and to model global lung anatomy.

### Why Vision Transformers?

- **Global Self-Attention.** Each patch can attend to every other patch, enabling the model to learn long-range dependencies (e.g., linking an opacity in the lower lobe to pleural changes elsewhere).  
- **Adaptive Feature Weighting.** Transformer layers dynamically re-weight contributions from different regions, which helps when disease signs vary in size, shape, and location.  
- **Scalability.** ViT scales efficiently with data and model size: fine-tuning on our 5 GB training set often yields richer representations than fixed-receptive-field CNNs.  
- **Medical Imaging State-of-the-Art.** Properly pre-trained and fine-tuned ViTs match or exceed CNN performance on chest X-ray tasks.

### Our Custom ViT Architecture

- **Convolutional Stem.**  
  - Three 3×3 Conv → BN → ReLU blocks, followed by two stride-2 convs, downsampling the input by 4× in each spatial dimension.  
  - Preserves local edge and texture features, while reducing sequence length (and quadratic attention cost) by ≈16×.  
- **Patch Embedding.**  
  - Flatten the feature H/4 * W/4 map into tokens, each projected to a D-dimensional embedding.  
  - Learnable positional embeddings and a **[CLS]** token for global classification.  
- **Transformer Backbone.**  
  - **Depth:** `depth` layers; **Heads:** `heads`; **MLP dim:** `mlp_dim`.  
  - Pre-LayerNorm + residual connections for stable training.  
  - Feed-forward MLP with dropout for regularization.  
- **Classification Head.**  
  - LayerNorm → Linear(\(D\), number_of_classes).  
- **Model Footprint.**  
  - ~5 M parameters, ~50 MB `.pth` file.

### Training Strategy & Experiment Tracking

- **Data Splits.** 5 GB train / 2 GB val / 1 GB test.  
- **Distributed Data-Parallel (DDP).**  
  - Multi-GPU gradient averaging yields near-linear speedups and smoother loss curves.  
  - Larger effective batch size stabilizes training and often improves accuracy.  
- **MLflow Logging.**  
  - Track per-epoch metrics (accuracy, loss, class-wise precision/recall).  
  - Log GPU utilization and learning-rate schedules.  
  - Artifact versioning of model checkpoints and hyperparameters for reproducibility.  
- **Ray Train.**  
  - Uses `RayDDPStrategy` to scale Lightning training seamlessly across a Ray cluster.  
  - Autoscaling of CPU/GPU resources and dynamic task scheduling for efficient utilization.  
  - Fault-tolerant execution with automatic recovery from worker node failures.

---

## Training Set up

### Environment Setup

This section describes how to provision a two-GPU VM on Chameleon Cloud and prepare it for ML training.

- **Configure the Chameleon context:**

```bash
context.version = "1.0"
context.choose_project()
context.choose_site(default="CHI@TACC")
```

- **Retrieve your GPU lease:**

```bash
l = lease.get_lease("project42_node")
l.show()
```

The VM instance detail is shown in Figure 1 below:

![VM](./Training_part/Image_Saved/VM.png)

- **Create and launch the VM:**

```bash
s = server.Server(
    name=f"node-mltrain-{os.getenv('USER')}",
    reservation_id=l.node_reservations[0]["id"],
    image_name="CC-Ubuntu24.04-hwe"
)
s.submit(idempotent=True)
s.associate_floating_ip()
s.refresh()
s.check_connectivity()
```

- **Install software prerequisites:**

```bash
s.execute("git clone https://github.com/.../MLOP-Med_Image_Pred")
s.execute("curl -sSL https://get.docker.com/ | sudo sh")
s.execute("amdgpu-install -y --usecase=dkms")
s.execute("sudo reboot")

s.refresh()
s.check_connectivity()
s.execute("rocm-smi")
s.execute("sudo apt -y install cmake libncurses-dev ...")
s.execute("git clone https://github.com/Syllo/nvtop")
s.execute("cd nvtop/build && cmake .. -DAMDGPU_SUPPORT=ON && sudo make install")
```

- **Build the ML Docker image:**

```bash
s.execute(
  "docker build -t jupyter-mlflow "
  "-f MLOP-Med_Image_Pred/Training_part/Dockerfile.jupyter-torch-mlflow-rocm ."
)
```

- **SSH into your VM:**

```bash
ssh -i ~/.ssh/key cc@<FLOATING_IP>
```

SSH login is shown in Figure 2 below:

![SSH Log in VM](./Training_part/Image_Saved/SSH.png)

### Data Preparation

- **Install and configure `rclone`:**

```bash
curl https://rclone.org/install.sh | sudo bash
sudo sed -i '/^#user_allow_other/s/^#//' /etc/fuse.conf
```

- **Create mount point and set permissions:**

```bash
sudo mkdir -p /mnt/object
sudo chown cc:cc /mnt/object
```

- **Configure the remote:**

```bash
[chi_tacc]
type = swift
user_id = f7aec218002617a11e8e21ff0ec3fe24c34a924443364a5d1c089c3048160669
application_credential_id = 9f60f8d9cfe144bc87dae88ebdb80a53
application_credential_secret = F4ASJEIbRKzLvhI_amIDYD6fStxhS7ABjAhzlgwTCGpv0AXx3qwbcvpDLoKd11sL7OiaalMb7aotsYebBGoHcQ
auth = https://chi.tacc.chameleoncloud.org:5000/v3 
region = CHI@TACC
```

- **List the Storage:**
  
```bash
rclone lsd chi_tacc:
```

![Reclone](./Training_part/Image_Saved/reclone.png)

- **Mount the bucket:**

```bash
rclone mount chi_tacc:object-persist-project42 \
      /mnt/object \
      --read-only \
      --allow-other \
      --daemon
```


Dataset structure:

![Directory structure](./Training_part/Image_Saved/Storage.png)

## Training Code Explanation

### Core Model and Training Logic

- **Imports and Hyperparameters**: PyTorch, einops, PyTorch Lightning.
- **Model Definition**:
  - `conv_stem`: Conv2d → BatchNorm → ReLU
  - `Transformer`: multi-head attention + feed-forward layers
  - `mlp_head`: LayerNorm → Linear
- **Data Loaders**: `get_dataloaders()`
- **LightningModule**:
  - `training_step`, `validation_step`, `test_step`
  - `configure_optimizers`: Adam optimizer
- **Trainer Launch**:

```python
Trainer(devices=2, accelerator="gpu", strategy=DDPStrategy, max_epochs=12, precision="bf16-mixed")
```

### MLflow Tracking

```python
mlflow.set_tracking_uri("http://129.114.27.23:8000")
mlflow.set_experiment("classifier")
mlflow.autolog(log_models=False)

mlflow.end_run()
mlflow.start_run(log_system_metrics=True)
mlflow.log_params(hparams)

info = subprocess.check_output("rocm-smi", shell=True)
mlflow.log_text(info.decode(), "gpu-info.txt")

if trainer.global_rank == 0:
  setup_mlflow(hparams)
  trainer.fit(...)
  trainer.test(...)
  mlflow.end_run()
```

### Ray Distributed Training

**Lightning with Ray DDP:**

```python
trainer = Trainer(
  strategy=RayDDPStrategy(),
  plugins=[RayLightningEnvironment()],
  devices="auto", accelerator="gpu", precision="bf16-mixed"
)
trainer = rlt.prepare_trainer(trainer)
trainer.fit(...)
trainer.test(...)
```

**Ray config:**

```python
from ray.train.torch import TorchTrainer, ScalingConfig, RunConfig, CheckpointConfig

run_config = RunConfig(
  storage_path="s3://ray",
  checkpoint_config=CheckpointConfig(
    checkpoint_score_attribute="val_loss",
    num_to_keep=3
  )
)
scaling_config = ScalingConfig(
  num_workers=2,
  use_gpu=True,
  resources_per_worker={"GPU": 1, "CPU": 8}
)
```

**TorchTrainer:**

```python
def train_func(config):
  # build model, loaders, and call trainer.fit here


TorchTrainer(
  train_loop_per_worker=train_func,
  scaling_config=scaling_config,
  run_config=run_config
).fit()

```
---

## Training Work UI

### MLflow Setup

This subsection describes how to launch a Jupyter container with MLflow UI, start the MLflow backend, and run the ViT training script with MLflow tracking.

1. **Check Datasets:**

```bash
cd work
ls /mnt/object
# → final_eval  merged_dataset  test  train  val
```

2. **Start MLflow backend:**

```bash
docker-compose -f docker-compose-mlflow.yaml up -d
```

This brings up the MLflow tracking server (with its database and UI) in detached mode.

3. **Launch Jupyter with MLflow UI:**
In NVIDIA:
```bash
docker run -d --rm \
  -p 8888:8888 \
  --gpus all \
  --shm-size 16G \
  -v ~/MLOP-Med_Image_Pred/Training_part:/home/jovyan/work/ \
  -v /mnt/object:/mnt/object \
  -e MLFLOW_TRACKING_URI=http://129.114.27.23:8000/ \
  --name jupyter \
  jupyter-mlflow
```
or in AMD:
```bash
docker run -d --rm \
  -p 8888:8888 \
  --device=/dev/kfd
  --device=/dev/dri
  --group-add video
  --group-add $(getent group render | cut -d: -f3)
  --shm-size 16G \
  -v ~/MLOP-Med_Image_Pred/Training_part:/home/jovyan/work/ \
  -v /mnt/object:/mnt/object \
  -e MLFLOW_TRACKING_URI=http://129.114.27.23:8000/ \
  --name jupyter \
  jupyter-mlflow
```
**Navigate to:**

```http
http://<your_float_ip>:8888/lab?token=<generated_token>
```

4. **Prepare code and run training:**

```bash
git clone https://github.com/ZBW-P/MLOP-Med_Image_Pred.git
cd MLOP-Med_Image_Pred
git switch -c mlflow

pip install einops

git config --global user.email "qh2262@nyu.edu"
git config --global user.name  "Qin Huai"
git add VIT_Mlflow.py
git commit -m "Apply MLflow tracking"
git log -n 2

python3 VIT_Mlflow.py
```

5. **Download trained checkpoint:**

```bash
mlflow artifacts download \
  --artifact-uri "runs:/<RUN_ID>/checkpoint.pth" \
  --dst-path ./model_ckpt

import torch
model = torch.load("model_ckpt/checkpoint.pth")
```
The Mlflow UI is shown below:

![Mlflow UI](./Training_part/Image_Saved/MLflow_UI_complete.png)

### Ray Cluster Configuration

This subsection explains how to build and launch a ROCm-enabled Ray cluster, set up a Jupyter client container, and submit distributed training jobs.

1. **Build and launch the ROCm Ray cluster:**

```bash
rocm-smi
docker build -t ray-rocm:2.42.1 \
  -f MLOP-Med_Image_Pred/Training_part/Dockerfile.ray-rocm .

docker-compose -f MLOP-Med_Image_Pred/Training_part/docker-compose-ray-rocm.yaml up -d
docker ps
docker exec ray-worker-0 rocm-smi
docker exec ray-worker-1 rocm-smi
```

2. **Build and run Jupyter client for Ray:**

```bash
docker build -t jupyter-ray \
  -f MLOP-Med_Image_Pred/Training_part/Dockerfile.jupyter-ray .

docker run -d --rm \
  -p 8888:8888 \
  -v ~/MLOP-Med_Image_Pred/Training_part:/home/jovyan/work \
  -v /mnt/object:/mnt/object \
  -e DATA_PATH=/mnt/object \
  -e RAY_ADDRESS="http://${HOST_IP}:8265" \
  --mount type=bind,source=/mnt/object,target=/mnt/object,readonly \
  --name jupyter \
  jupyter-ray
```

**Access Jupyter at:**

```http
http://<your_float_ip>:8888/lab?token=<token>
```

3. **Prepare runtime environment:**

- **requirements.txt:**

```bash
torchvision
einops
lightning
torch
```

- **runtime.json:**

```json
{
  "pip": "requirements.txt",
  "env_vars": {
    "DATA_PATH": "/mnt/object"
  }
}
```

Clone and switch to the Ray branch:

```bash
git clone https://github.com/ZBW-P/MLOP-Med_Image_Pred.git
cd MLOP-Med_Image-Pred
git switch -c ray
git rm train.py
git commit -m "Remove train.py—use VIT.py as entrypoint"
cp VIT_Ray.py gourmetgram-train/
cd gourmetgram-train
git add VIT_Ray.py
git commit -m "Ensure VIT.py uses DATA_PATH"
```

4. **Submit distributed training job:**

```bash
ray job submit \
  --runtime-env runtime.json \
  --working-dir . \
  -- python MLOP-Med_Image_Pred/train.py
```

The Ray UI is shown below:

![Ray UI](./Training_part/Image_Saved/Ray_UI.png)


---

## Experimental Results

### Mlflow DDP Outcome

The Mlflow Training is shown in Figure below:

![Mlflow Training](./Training_part/Image_Saved/Mlflow_training.png)

The graph of the Mlflow training completion is shown in Figure below:

![UI for Mlflow training completed](./Training_part/Image_Saved/Mlflow_complete.png)

The Mlflow training GPU usage is shown in Figure below:

![Training Resource usage](./Training_part/Image_Saved/Mlflow_resource_usage.png)

---

### Ray Train Outcome

The Ray train job trained successfully as shown in Figure below:

![Ray Train Job complete](./Training_part/Image_Saved/Ray_Train_Job.png)

The Ray train job resource usage is shown in Figure below:

![Ray Train Resources usage](./Training_part/Image_Saved/Ray_Train_Resources.png)

The Ray head and workers working status is shown in Figure below:

![Ray cluster](./Training_part/Image_Saved/Ray_cluster.png)

---

### Retrain Outcome

**User-selected Dataset Augmentation:**

To further test model robustness, we introduced an interactive dataset augmentation mechanism. Users specify the class (e.g., `lung-viral-pneumonia`) and select a 10% subset of unused images, which are integrated into the training, validation, and test datasets as example.

```python
def add(train_loader, val_loader, test_loader, batch_size, file_path, num_workers: int =16, ratio: float=0.1 ,seed: int = 42):
 """
    Augment existing DataLoaders with a new slice of images from disk.

    For a given `ratio` of the dataset in `file_path/final_eval`, this function:
      1. Loads the full ImageFolder dataset from `file_path/final_eval`.
      2. Picks a contiguous slice of size `ratio * dataset_size`, based on an implicit `offset` of 0–9.
      3. Shuffles that slice with `seed + offset`.
      4. Splits it into train/val/test subsets (70/20/10%).
      5. Concatenates those subsets onto the original `train_loader.dataset`, `val_loader.dataset`, and `test_loader.dataset`.
      6. Returns three new DataLoaders built with `batch_size` and `num_workers`.

    Args:
        train_loader (DataLoader): Original training DataLoader.
        val_loader (DataLoader):   Original validation DataLoader.
        test_loader (DataLoader):  Original test DataLoader.
        batch_size (int):          Batch size for the returned loaders.
        file_path (str):           Base path to your datasets.
        num_workers (int):         Number of workers for the new loaders.
        ratio (float):             Fraction (0–1) of the new dataset to add each call.
        seed (int):                Random seed base for reproducible shuffling.

    Returns:
        tuple: (new_train_loader, new_val_loader, new_test_loader)
    """
```

The Retrain terminal operation of adding lung-viral-pneumonia(10% of eval dataset) is shown in Figure below:
![Retrain 1](./Training_part/Image_Saved/Retrain_lung-viral-pneumonia.png)

The Retrain Mlflow record resources is shown in Figure below:
![Retrain 2](./Training_part/Image_Saved/Retrain_lung-viral-pneumoniaGPU.png)

The Retrain Mlflow UI training success is shown in Figure below:
![Retrain 3](./Training_part/Image_Saved/Retrain_mlflow_ui.png)

---

### Normal training Outcome

**Single-GPU Training:**

The normal (single-GPU) training scenario took 30.8 minutes for 12 epochs and achieved an accuracy of approximately 86%. The training exhibited stable but slower convergence compared to distributed approaches.

The Normal train with 1036 batch is shown in Figure below:
![Normal 1](./Training_part/Image_Saved/Normal.png)

The Normal train Mlflow record resources is shown in Figure below:
![Normal 2](./Training_part/Image_Saved/Normal_GPU_matrix.png)

The Normal train Mlflow UI training success is shown in Figure below:
![Normal 3](./Training_part/Image_Saved/Normal_finished_overview.png)

---

### Compare Normal & DDP-Strategy & Ray Train

**Comparative Analysis Across Strategies:**

The experimental comparisons highlight the performance, efficiency, and resource utilization across single-GPU training (Normal), Distributed Data Parallel (DDP), and Ray distributed training approaches.

The Comparsion with logs matrics is shown in Figure below:
![compare 1](./Training_part/Image_Saved/Comparison.png)

The Comparsion with parameters setting is shown in Figure below:
![compare 2](./Training_part/Image_Saved/Comparison_2.png)


| Strategy           | Accuracy (%) | Loss  | Training Time (12 epochs) |
|--------------------|--------------|-------|----------------------------|
| **Ray Train**      | 87.0         | 0.390 | 28.15 min                    |
| **Retrain DDP**    | 86.7         | 0.388 | 21.4 min                   |
| **Normal**         | 86.4         | 0.433 | 30.8 min                   |
| **Baseline DDP**   | 84.8         | 0.445 | 19.5 min                   |

---

### Discussion

**Mlflow & Ray Train:**

- The Mlflow DDP training exhibited rapid initial convergence, followed by gradual stabilization, achieving final accuracy of 85% and minimum validation loss of 0.44 by epoch 12.
- Ray Train surpassed this with higher accuracy (87%) and lower loss (0.39), benefiting from its distributed data handling and efficient parallel execution capabilities.

**Retrain Experiment:**

- Integrating an additional 10% of targeted class data improved both accuracy and loss metrics. This demonstrates the model's ability to leverage targeted dataset expansions effectively.

**Strategy Comparisons:**

- Ray Train DDP achieved the highest accuracy but requires infrastructure overhead and the training period is the highest.
- DDP (Retrain) offered the best balance between training efficiency, accuracy, and simplicity.
- Normal single-GPU training is viable but notably slower and slightly less accurate, reflecting limitations in batch sizes and computational throughput.



---

## Unit 6 & 7: Model Serving and Evaluation

**API Endpoint**  
- Script: [`app.py`](link)  
- Input format: image URL or file  
- Output format: JSON prediction
- ['http://129.114.27.23:8265/'](link)
- <img width="1018" alt="image" src="https://github.com/user-attachments/assets/95d4527c-5446-4c23-961b-689c2c42290f" />

**Customer Requirements**  
- Low-latency (<500ms), high-confidence, interpretable outputs

**Model Optimizations**  
- Tools: TorchScript, ONNX, etc.  
- Results: reduced inference latency or model size

**Offline Evaluation**  
- Test suite: [`tests/offline_eval.py`](link)  
- Metrics: accuracy, AUC, per-class F1, etc.  
- Latest results: [MLflow link](link)

**Load Test (Staging)**  
- Script: [`tests/load_test.py`](link)  
- Results: screenshot or link to logs/dashboard

**Business-Specific Metric**  
- Define: e.g. “% correct urgent diagnoses within 1 minute”  
- Reasoning behind the metric

**(Optional) Multi-Serving Options**  
- REST vs gRPC implementation and performance comparison  
- Cost analysis: link to cloud calculator or spreadsheet

---

## Unit 2/3: Staged Deployment

**Deployment Flow**  
- Staging → Canary → Production  
- Tracking tool: MLflow versioning or GitHub tags  
- Script/CI: [`ci/deploy_pipeline.yml`](link)

---

## Unit 8 (Online): Real-Time Data and Feedback

**Online Data Flow**  
- Real-time inputs pushed via: [`data/send_live.py`](link)  
- Storage: MinIO “inference-inputs” bucket  
- Connected to FastAPI service

---

## Unit 6 & 7: Online Evaluation & Monitoring

**Monitoring and Evaluation**  
- Monitoring setup: Prometheus + Grafana  
- Dashboard: [Grafana link](http://your-server:3000)

**Feedback Loop**  
- Online labels (if applicable) collected via: [`feedback/collect_labels.py`](link)  
- Used to trigger re-training or evaluation

**(Optional) Data Drift Monitoring**  
- Code: [`monitoring/check_drift.py`](link)  
- Visualization: Grafana dashboard

**(Optional) Model Degradation Monitoring**  
- Accuracy/AUC tracked over time  
- Alerts set in Prometheus

---

## Unit 2/3: CI/CD and Continuous Training

**Continuous Pipeline**  
- New data in MinIO triggers GitHub Action  
- Script: [`ci/continuous_train.yml`](link)  
- End-to-end: Data → Retrain → Deploy → Monitor


