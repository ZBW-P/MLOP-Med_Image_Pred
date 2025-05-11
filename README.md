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
Insert infrastructure diagram here (linked or embedded).  
Example:  
`![Architecture](link/to/diagram.png)`

**Infrastructure-as-Code**  
- Setup scripts: [`infra/setup.sh`](link)  
- Compose file: [`infra/docker-compose.yml`](link)  
- Kubernetes: [`k8s/`](link)

---

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

**Modeling Problem Setup**  
- Inputs: e.g. 256x256 grayscale images  
- Outputs: e.g. multi-label vector  
- Model: Vision Transformer (ViT) with customized encoder  
- Why this model: suitability to limited-labeled medical data

**Training Process**  
- Script: [`train.py`](link)  
- Re-training: [`scripts/retrain.py`](link)

**Experiment Tracking**  
- Tool: MLflow  
- Dashboard: [Link to MLflow](http://your-server:5000)  
- Comparison of runs: screenshot or link

**Training Scheduler / CI Integration**  
- Example YAML: [`ci/train_trigger.yaml`](link)

**(Optional) Advanced Training**  
- DDP / FSDP / Deepspeed usage: details and benchmark improvements  
- Ray Tune: link to relevant configs/code

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


