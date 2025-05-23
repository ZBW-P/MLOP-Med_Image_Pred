## MLOP based on VIT

### Project Overview

Our project develops a **hybrid machine learning system** that integrates a Vision Transformer (ViT) for initial disease classification from chest X-ray images with a Large Language Model (LLM) for interpretability and actionable insights. In current clinical settings, radiologists manually assess chest X-rays—a process that is time-consuming, subjective, and prone to inconsistency. Our system addresses these issues while satisfying the project’s requirements for scale in data, model complexity, and deployment.

### Key Value Propositions

#### 1. User Preview and Enhanced Communication
- **Preliminary Results:** Provides users with an immediate preliminary assessments of potential chest abnormalities from their X-ray images.
- **Improved Communication:** Helps patients articulate their symptoms more effectively during consultations, especially while in the waiting room.
- **Fast, Accessible Feedback:** Thanks to the model’s efficient design and quick update mechanism, users receive real-time preliminary assessments. Although the model may trade off some accuracy compared to professional-grade systems, its speed and ease-of-access allow patients to obtain useful insights promptly.

#### 2. Reduction of Patient Revisits
- **Initial Prediction:** Offers an early indication of whether findings are mild or severe, which can reduce the need for multiple doctor visits (reducing the cost per visit).
- **Feedback Loop:** Physician-confirmed feedback is incorporated to refine future predictions, ensuring that the system’s recommendations remain safe and reliable.

#### 3. Lightweight and Accessible Deployment
- **Balanced Performance:** Our model is designed to be medium-sized, striking a balance between computational efficiency and performance. This enables easy deployment on websites and mobile applications without requiring extensive hardware resources.
- **Multi-Platform Use:** Its lightweight nature means it can be integrated into various devices, ensuring that users have quick access to preliminary predictions.
- **Efficient Model Size & Storage:** Unlike professional models that are large and resource-intensive, our system is optimized for lower storage needs and faster computation. This makes it easier to update and maintain, while still delivering valuable insights.

#### 4. Automated Feedback and Model Upgrades
- **Continuous Improvement:** User interactions and professional feedback are automatically managed via our dedicated web platform and server, creating a self-improving system.
- **Model Supervision:** Ongoing monitoring and regular upgrades ensure that the model remains clinically relevant over time.
- **Optimized Update Mechanism:** By leveraging an online server infrastructure for both image storage and model updates, the system maintains a balanced trade-off between model size and accuracy. This allows patients, who might otherwise face repeated doctor visits, to benefit from timely and cost-effective preliminary assessments.

#### Disclaimer

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

### Detailed design plan

<!-- In each section, you should describe (1) your strategy, (2) the relevant parts of the 
diagram, (3) justification for your strategy, (4) relate back to lecture material, 
(5) include specific numbers. -->

### Model training and training platforms


### Model serving and monitoring platforms

<!-- Make sure to clarify how you will satisfy the Unit 6 and Unit 7 requirements, 
and which optional "difficulty" points you are attempting. -->
Our Model Serving component supports both **real-time (online)** and **batch (offline)** inference modes for a hybrid ML system that combines a **Vision Transformer (ViT)** for disease classification with a **Large Language Model (LLM)** for diagnostic explanations.

- **Serving Requirements:** 
The model will be encapsulated within a RESTful API to facilitate real-time predictions.

Online Inference: Low latency is essential for interactive, single-sample predictions.
Batch Inference: High throughput is required to process large datasets efficiently.

Lightweight models may be deployed on edge devices to minimize latency, while more complex models could be deployed in the cloud to achieve higher accuracy.

- **Model-Level Optimizations:** 
To optimize the model performance, we will use several strategies:

For graph optimizations, we could convert the model to ONNX to enable operation fusion to reduce redundant computations.

For dynamic quantization, only the weights are quantized in advance (stored as INT8), while the quantization parameters for activations are computed during inference.

For static quantization, both weights and activations are quantized in advance. A calibration dataset is used to compute the quantization parameters for activations.

- **System-Level Optimizations:**
We would implement warm start mechanisms to pre-load models, ensuring minimal latency for the initial requests, and deploy multiple instances of the model to handle a high volume of concurrent requests.

### Evaluation and Monitoring
- **Offline Evaluation：**
After model training, an automated offline evaluation pipeline will immediately run and log results to MLFlow. This evaluation will include:
  1. **Standard and Domain-Specific Use Cases:** Testing on both general diagnostic scenarios and specialized medical contexts to ensure performance.
  2. **Known Failure Mode Testing:** Running tests on predetermined failure cases to verify the model’s resilience in edge cases.
  3. **Template-Based Unit Tests:** Automatically executing unit tests based on established templates to validate key functionalities.

**Staging Load Testing**

Once the Continuous X pipeline deploys the service to a staging environment, a load test will be conducted. This test will simulate high-concurrency conditions to evaluate the system’s response time, throughput, and overall stability.

**Canary Online Evaluation**

After passing staging tests, the service will be deployed to a canary environment. This phase will assess the system’s response times, stability, and diagnostic accuracy under conditions that mimic actual user interactions.

**Business-Specific Evaluation**

Although the system is not yet fully deployed to production, a business-specific evaluation plan has been defined. Key business metrics—such as diagnostic accuracy, misdiagnosis rates, response times, and service availability—will be tracked via MLFlow and integrated monitoring platforms.

**Data drift monitoring**

We will determine and monitor whether data drift has occurred based on changes in the model’s accuracy and recall.

### Data pipeline

#### Overview
The data pipeline is designed to support both offline and online data processing for our hybrid ML system integrating a Vision Transformer (ViT) for disease classification and a Large Language Model (LLM) for interpretability. It includes infrastructure for persistent storage, ETL processing, simulated data streaming, and an optional interactive dashboard.
#### Persistent Storage
We provision persistent storage on Chameleon Cloud, using:

- **Docker Volumes** for intermediate storage.
- **MinIO** (deployed via Docker Compose) for object storage of unstructured data (X-ray images, model artifacts).
- **PostgreSQL** (self-hosted via Docker Compose) for structured data like labels, metadata, and logs.

MLflow will be used for experiment tracking:
- **Backend:** PostgreSQL (mlflowdb)
- **Artifact Store:** MinIO (bucket: mlflow-artifacts)

---

#### Offline Data Management

**Example Data Sources:**
- COVID-QU-Ex dataset:
  - 11,956 COVID-19 images
  - 11,263 Non-COVID infections
  - 10,701 Normal cases
  - Lung_Opacity cases with metadata (age, gender, diagnosis)

**Repository Structure:**
- **Structured data:** PostgreSQL (labels, metadata)
- **Unstructured data:** MinIO (X-ray images, models)

**ETL Pipeline:**
- **Extract:** Load raw images and metadata
- **Transform:**
  - Remove corrupted images
  - Normalize brightness/contrast
  - Augment data: rotation, flipping, etc.
  - Convert categorical metadata
- **Load:**
  - Upload processed data to MinIO
  - Store labels and metadata in PostgreSQL
  - Log data and models in MLflow

**Support for Re-training:**
- Monitor MLflow logs and user feedback in PostgreSQL
- Re-run ETL for flagged samples
- Feed data back into training loop

---

#### Online Data Management

**Streaming Pipeline:**
- Handle real-time image uploads via a REST API
- Process images and send them to:
  - ViT for classification
  - LLM for diagnosis explanation

**ETL Pipeline for Online Data:**
- **Extract:** Receive images and metadata from frontend/API
- **Transform:** Resize, normalize, convert format (DICOM → PNG)
- **Load:**
  - Store results (label + explanation) in PostgreSQL
  - Store cleaned images in MinIO
  - Log run metadata in MLflow

---

#### Simulating Online Data

Since real user data is not available:

**Simulation Script (Python):**
1. Randomly sample from offline datasets
2. Mimic API uploads at set intervals
3. Monitor results, store in PostgreSQL
4. Raise alerts if model performance drops

Characteristics:
- Interval-based requests
- Diverse image types (COVID, Pneumonia, etc.)
- Metadata attached to simulate realistic submissions



### Continuous X 

#### **1. Overview**

The Continuous X pipeline implements DevOps best practices to automate
our machine learning system's lifecycle. This comprehensive solution
ensures reliable, scalable, and reproducible operations from development
through production deployment.

Key Components:

-   Infrastructure-as-Code (IaC) provisioning

-   Cloud-native microservices architecture

-   Automated CI/CD workflows

-   Staged deployment environments

-   Comprehensive monitoring

#### **2. Infrastructure-as-Code Implementation**

##### **2.1 Tooling Stack**

 
| Component   |       Technology|   Purpose|
|-----------|------------------|------|
|Provisioning       |Terraform   | Cloud resource management|
|Configuration      |Ansible     | System setup automation|
|Containerization   |Docker      | Service packaging|
|Orchestration      |Kubernetes   |Production deployment|


##### **2.2 Key Features**

-   Version-controlled infrastructure definitions

-   Automated environment provisioning

-   Immutable infrastructure pattern

-   Repeatable configuration management

#### **3. Cloud-Native Architecture**

##### **3.1 Service Breakdown**

  
  |Service         |Technology Stack      | Responsibility|
  |---|---------------------------------|-------|
  |ViT Model API   |FastAPI + PyTorch      |Image classification|
  |LLM Service     |FastAPI + LitGPT       |Diagnostic explanations|
  |Monitoring      |Prometheus + Grafana  | Performance tracking|
  |Data Pipeline   |Airflow + MinIO        |Data processing|


##### **3.2 Container Strategy**

-   GPU-optimized containers for ML workloads

-   Lightweight Alpine-based containers for services

-   Helm charts for Kubernetes deployments

#### **4. CI/CD Pipeline**

##### **4.1 Pipeline Stages**

1.  **Code Commit**

    -   Trigger: Push to main branch

    -   Validation: Linting, unit tests

2.  **Training Phase**

    -   Distributed training on Ray cluster

    -   Experiment tracking via MLFlow

3.  **Evaluation Gate**

    -   Automated model validation

    -   Fairness and bias testing

4.  **Packaging**

    -   Docker image creation

    -   Artifact storage in registry

5.  **Staged Deployment** Progressive rollout strategy

##### **4.2 Quality Gates**


  |Checkpoint             |Success Criteria    |Failure Action|
  |--|--|--|
  |Model Accuracy         |\>70% on test set   |Halt pipeline|
  |Inference Latency      |\<300ms P99         |Rollback|
  |Resource Utilization   |\<80% GPU memory    |Scale up resources|


#### **5. Deployment Strategy**

##### **5.1 Environment Matrix**


  |Environment   |Purpose                 |Traffic   |Validation Method|
  |--|--|--|--|
  |Development   |Feature testing         |0%        |Manual verification|
  |Staging       |Integration testing     |0%        |Automated load testing|
  |Canary        |Real-world simulation   |10%       |A/B testing|
  |Production    |Live service            |90%       |Continuous monitoring|


##### **5.2 Rollback Protocol**

-   Automatic triggers:

    -   Latency \>500ms for 5 minutes

    -   Error rate \>1% sustained

    -   Data drift detected

-   Manual override capability

-   Versioned rollback targets

#### **6. Monitoring & Observability**

##### **6.1 Monitoring Stack**

-   **Metrics Collection:** Prometheus

-   **Visualization:** Grafana

-   **Logging:** ELK Stack

-   **Model Tracking:** MLFlow

##### **6.2 Key Dashboards**

1.  System Health

    -   Resource utilization

    -   Service availability

2.  Model Performance

    -   Inference latency

    -   Prediction accuracy

    -   Data drift metrics

3.  Business Impact

    -   Diagnostic outcomes

    -   Physician feedback
