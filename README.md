# Project Title: [Your Project Name]

## Unit 1: Value Proposition and Scale

**Target Customer**  
Describe the *one specific* customer you are designing for.  
Example: “We are building a tool for radiologists in community hospitals without AI specialists.”

**Value Proposition**  
Briefly describe how your system helps this specific customer.  
Example: “Our system allows quick diagnosis of X-ray images with visual explanations.”

**Scale**  
- Data size: ~[XX] GB  
- Model training time: ~[XX] hours on [hardware]  
- Inference volume: ~[XX] requests per day/hour

---

## Unit 2/3: Cloud-Native Infrastructure

**Architecture Diagram**  
![infrastructure](https://github.com/user-attachments/assets/1b30abd6-74c2-4c73-8b6a-ab322f033f0a)


We provisioned resources on **Chameleon Cloud (KVM@TACC)** using the **OpenStack CLI**, and configured services via the **Jupyter interface** by running [`infrastructure_provision/provision_cli.ipynb`](infrastructure_provision/provision_cli.ipynb).

### Provisioning Summary

- **Private Network**: `private-net-project42`
- **Nodes**:
  - `node1-mlops-project42-1` (192.168.1.11 / public: 129.114.27.23)
  - `node2-mlops-project42-1` (192.168.1.12)
  - `node3-mlops-project42-1` (192.168.1.13)
- **Flavor**: `m1.xlarge` (8 vCPUs, 16GB RAM, 40GB disk)
- **Image**: `CC-Ubuntu24.04`
- **Volume attached**: `block-persist-project42-1` on `/dev/vdb`

### Instance Configuration & Setup

- Infrastructure is initialized via `config-hosts.yaml`
- All nodes were launched using `openstack server create --port --user-data ...`

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

**Persistent Storage**  
List and describe each bucket/volume:  
- `minio/production/raw-data/`: raw input files  
- `minio/production/processed/`: normalized and split datasets  
Include disk usage if possible.

**Offline Dataset**  
- Datasets used: Chest X-ray, OCT, etc.  
- Scripts: [`data/prepare_data.py`](link)  
- Sample illustration: insert or link representative samples

**Data Pipeline**  
- Source → Preprocessing → Object Storage  
- Splitting strategy: train/val/test/final_eval (mention person ID integrity if applicable)  
- Data cleaning steps (resizing, normalization, etc.)

**(Optional) Data Dashboard**  
- Location: [`dashboards/data_dashboard.ipynb`](link)  
- Insight example: class imbalance, missing fields

---

## Unit 4 & 5: Modeling, Training, and Experiment Tracking

[See the full README](./Training_part/Image_Saved/readme.md)

---

## Unit 6 & 7: Model Serving and Evaluation

**API Endpoint**  
- Script: [`serve/inference_api.py`](link)  
- Input format: image URL or file  
- Output format: JSON prediction

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


