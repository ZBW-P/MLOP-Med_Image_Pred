## Title of project

Our group is devoted to developing a hybrid machine learning system that integrates a Vision Transformer (ViT) for disease classification with a Large Language Model (LLM) for interpretability and actionable insights. In many clinical settings today, diagnoses from medical images such as chest X-rays are performed manually by radiologists—a process that is time-consuming, subjective, and inconsistent.

By automating the initial classification, our system leverages the ViT to analyze medical images and produce output scores that reflect the likelihood of various diseases. The LLM then interprets these scores by comparing them against established benchmarks, providing patients with an early, preliminary understanding of their condition—indicating whether their situation is mild or severe—and offering insights into what they may face.

Moreover, this approach serves as an effective first-line screening tool for physicians. Doctors can use the system’s preliminary assessments to determine if more detailed examinations are necessary, which helps reduce misdiagnosis and improves overall diagnostic accuracy. This, in turn, enhances treatment efficiency and optimizes hospital workflows.

Our system's success will be evaluated based on improved diagnostic metrics (e.g., accuracy, sensitivity, and specificity) as well as reduced time-to-decision in clinical workflows.

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
| Zhaochen Yang                  |Model serving                 |https://github.com/ZBW-P/MLOP-Med_Image_Pred/blob/main/README.md#model-serving-and-monitoring-platforms                                    |
| Junjie Mai                   |Data Pipeline                 |                                    |
| Team member 4 (if there is one) |                 |                                    |



### System diagram

<!-- Overall digram of system. Doesn't need polish, does need to show all the pieces. 
Must include: all the hardware, all the containers/software platforms, all the models, 
all the data. -->

### Summary of outside materials

<!-- In a table, a row for each dataset, foundation model. 
Name of data/model, conditions under which it was created (ideally with links/references), 
conditions under which it may be used. -->

Our overall approach is to leverage the dataset detailed in the table below as the foundation for our ML operations system. Our pipeline integrates persistent storage, device-based processing, and Docker orchestration to streamline training, evaluation, and validation. Specifically, we plan to:

- **Centralize Data Storage:**  
  Store the collected chest X-ray images in a clear and organized library on persistent storage. The data will be segmented into three zipped folders (`train`, `test`, and `val`), ensuring that all training, evaluation, and testing sets are easily accessible to our system.
  
- **Model training and evaluation:**
  We will train a custom Vision Transformer (ViT) model on chest X-ray images for accurate medical image classification. A LitGPT-derived LLM will be employed to generate diagnostic derivations and actionable recommendations based on the outputs of the ViT model. Our training pipeline will support continuous training, evaluation, and re-training on new or updated data to ensure that both models remain current and adapt to evolving clinical requirements.
    
- **Integration with Our Pipeline and Device Infrastructure:**  
  Use Python scripts to read and preprocess images from the persistent storage. These scripts will run on dedicated devices (e.g., GPU-enabled servers) that are integrated into our Docker-based ML operations system. The images will be resized and formatted as needed based on our model requirements, ensuring that data is properly prepared for ingestion into our training pipeline.


The table below summarizes the datasets we plan to use, including details on how each dataset was created and the conditions under which it may be used.

| Name of Data/Model                              | How it was Created                                                                                                                                                                                                                                                                                                                                                                                                                 | Conditions of Use                                                                                                                                                                                                                                                       |
|-------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Chest X-Ray Images (Pneumonia)**             | This dataset comprises 5,863 JPEG chest X-ray images organized into three folders (train, test, val) with two subcategories (Pneumonia/Normal). The images were retrospectively collected from pediatric patients (ages 1–5) at Guangzhou Women and Children’s Medical Center. All radiographs underwent quality control by removing unreadable scans, and diagnoses were graded by two expert physicians (with a third review for evaluation). Refer to [Kaggle Dataset Page](https://www.kaggle.com/code/madz2000/pneumonia-detection-using-cnn-92-6-accuracy/input). | Licensed under CC BY 4.0[Licenses](https://creativecommons.org/licenses/by/4.0/). The dataset is available for academic and non-commercial research. Proper citation of the original article and dataset is required. |
| **COVID-19 Radiography Database**              | Developed by a collaborative team from Qatar University, the University of Dhaka, and partner institutions, this dataset aggregates chest X-ray images for COVID-19, Normal, and Viral Pneumonia cases. The initial release provided 219 COVID-19, 1341 Normal, and 1345 Viral Pneumonia images, with updates increasing the COVID-19 cases to 3616 and including corresponding lung masks. More details can be found on the [Kaggle Dataset Page](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/data). | Released for academic and non-commercial use. Users must cite the original publications (Chowdhury et al., IEEE Access, 2020; Rahman et al., 2020[Publication1](https://ieeexplore.ieee.org/document/9144185),[Publication2](https://www.sciencedirect.com/science/article/pii/S001048252100113X?via%3Dihub) and adhere to the dataset usage guidelines provided in its metadata.                                                                     |
| **LitGPT-based LLM for Diagnostic Derivation**      | Developed by Lightning AI and built from scratch—drawing on innovations from nanoGPT, GPT-NeoX, bitsandbytes, LoRA, and Flash Attention 2—this LLM is designed to provide fast, minimal, and high-performance inference at enterprise scale. In our system, the model will be used in conjunction with our custom deep learning image classifier to analyze medical images and derive diagnostic outcomes along with actionable recommendations. For further details, refer to the [LitGPT Quick Start Guide](https://github.com/Lightning-AI/litgpt#quick-start). | Licensed under the Apache License 2.0, which permits unlimited enterprise use, modification, finetuning, pretraining, and deployment. Users are required to provide proper citation (e.g., see [LitGPT citation](https://github.com/Lightning-AI/litgpt)) when using or extending the model.         |
| **Tuberculosis (TB) Chest X-ray Database** | Developed by a collaborative research team from Qatar University, University of Dhaka, Malaysia, and affiliated medical institutions, this database comprises chest X-ray images for TB-positive cases and Normal images. The current release includes 700 publicly accessible TB images, 2800 additional TB images available via a data-sharing agreement through the NIAID TB portal, and 3500 Normal images. The dataset is compiled from multiple sources (including the NLM, Belarus, NIAID TB, and RSNA CXR datasets) and was used in the study “Reliable Tuberculosis Detection using Chest X-ray with Deep Learning, Segmentation and Visualization” published in IEEE Access. For further details, refer to the [Kaggle Dataset Page](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset). | Licensed for academic and non-commercial research use. Users must provide proper citation to the original publication: Tawsifur Rahman, Amith Khandakar, Muhammad A. Kadir, Khandaker R. Islam, Khandaker F. Islam, Zaid B. Mahbub, Mohamed Arselene Ayari, Muhammad E. H. Chowdhury. (2020) “Reliable Tuberculosis Detection using Chest X-ray with Deep Learning, Segmentation and Visualization”. IEEE Access, Vol. 8, pp 191586–191601 (DOI: 10.1109/ACCESS.2020.3031384), and adhere to the dataset usage guidelines. |
| **Bone Fracture Detection** | A comprehensive dataset of X-ray images created specifically for automated bone fracture detection in computer vision projects. The dataset includes images categorized into various classes (Elbow Positive, Fingers Positive, Forearm Fracture, Humerus Fracture, Shoulder Fracture, and Wrist Positive), with each image annotated using bounding boxes or pixel-level segmentation masks to indicate the location and extent of fractures. Designed to facilitate the development and evaluation of robust object detection models in medical diagnostics, the dataset accelerates research in automated fracture detection. For further details, refer to the [Kaggle Dataset Page](https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project). | Licensed under Attribution 4.0 International (CC BY 4.0). When using this dataset, please cite it by the DOI: 10.13140/RG.2.2.14400.34569 and reference the corresponding publication on [ResearchGate](https://www.researchgate.net/publication/382268240_Bone_Fracture_Detection_Computer_Vision_ProjectLicense). |


### Summary of infrastructure requirements

<!-- Itemize all your anticipated requirements: What (`m1.medium` VM, `gpu_mi100`), 
how much/when, justification. Include compute, floating IPs, persistent storage. 
The table below shows an example, it is not a recommendation. -->

| Requirement     | How many/when                                     | Justification |
|-----------------|---------------------------------------------------|---------------|
| `m1.medium` VMs | 3 for entire project duration                     | ...           |
| `gpu_mi100`     | 4 hour block twice a week                         |               |
| Floating IPs    | 1 for entire project duration, 1 for sporadic use |               |
| etc             |                                                   |               |

### Detailed design plan

<!-- In each section, you should describe (1) your strategy, (2) the relevant parts of the 
diagram, (3) justification for your strategy, (4) relate back to lecture material, 
(5) include specific numbers. -->

### Model training and training platforms

<!-- Make sure to clarify how you will satisfy the Unit 4 and Unit 5 requirements, 
and which optional "difficulty" points you are attempting. -->

#### Overview

- **Image Classification (ViT):**  
  Train a robust ViT model on a large-scale dataset (5+ GB) consisting of x-ray images categorized into 10 classes (e.g., pneumonia, fractures, and other diseases). This model is engineered to extract fine-grained features from complex medical images and provide initial diagnostic insights.

- **Derivation & Suggestion (LLM):**  
  Develop a 7B+ parameter LLM to analyze the output from the classification model and generate detailed suggestions. These suggestions serve as a secondary check, aiding doctors in their diagnosis and giving patients early pre-diagnostic insights.

---

#### Model Training Details

Medical Image Classification (ViT)

- **Dataset:**  
  - Over 5 GB of medical x-ray images classified into 10 distinct disease categories.

- **Architecture:**  
  - A large Vision Transformer (ViT) model tailored to capture the nuanced features of x-ray images.

- **Distributed Training Strategies:**  
  - **DDP (Distributed Data Parallel):** Every GPU computes the gradient for the entire model.  
  - **FSDP (Fully Sharded Data Parallel):** Each GPU computes only a portion of the outcome and gradient, improving memory efficiency.

- **Hardware Setup:**  
  - Utilizing 4 GPUs in parallel to accelerate training and manage the heavy computational load.

- **Training process:**  

  Our group plans to do the following training process:
  
    1. **Build a Docker Container:**  
          - Set up a Docker container with all the required resources, including the NVIDIA container toolkit. The most important component in our environment is the PyTorch library like PyTorch Lightning.
      
    2. **Implement Distributed Training Strategies:**  
          - To utilize both DDP and FSDP, our group will incorporate the PyTorch Lightning library. With PyTorch, we can configure the trainer using `DDPStrategy` and `FSDPStrategy` for effective model training.
      
    3. **Monitoring and Performance Tracking:**  
          - Use `nvtop` to monitor GPU usage and performance.
          - Employ `myflow` and Ray Training to track the model's performance under different training strategies.
      
    4. **Objective:**  
          - Deliver high-accuracy predictions that help researchers rapidly analyze images and assist doctors in making informed decisions. Even if the suggestions are not perfect, they provide a valuable second opinion in the diagnostic process.

Derivation & Suggestion (LLM)

- **Purpose:**  
  - Leverage a large language model to process the output from the ViT and generate clinical suggestions, enhancing both the efficiency and the accuracy of preliminary diagnoses.

- **Architecture:**  
  - An LLM with a minimum of 7B parameters designed to handle large-scale, detailed derivations.

- **Training Strategies:**  
  - Use of smaller batch sizes and gradient accumulation.  
  - Reduced precision training for efficiency.  
  - Parameter-efficient fine-tuning techniques such as LoRA and QLoRA, inspired by methods from our lab assignments.

- **Integration:**  
  - The refined parameters from the ViT model will be passed to the LLM to inform its derivation process, ensuring that the final output is both contextually relevant and actionable.

- **Training Process**：

  Our group plans to follow a similar approach as in the lab assignment:
  
    1. **Initial Testing:**  
      - Begin by testing the training speed for Reduced precision training and Gradient accumulation strategies depends on our model setup (May be larger).
  
    3. **Model Setup:**  
      - Define our LLM model with a minimal size (pre-trained) to meet initial requirements. The model will leverage:
        - Reduced precision training
        - Gradient accumulation
  
    4. **Potential Enhancements:**  
      - Define our LLM model with a minimal size (pre-trained) to meet initial requirements. The model will leverage:
      If improved accuracy is required for our service in the future, we plan to explore parameter-efficient fine-tuning techniques such as LoRA to further accelerate training and enhance the LLM’s derivation performance.
  
---

#### Experiment Tracking & Training Infrastructure

Experiment Tracking

- **MLFlow Integration:**  
  All experiments are tracked using MLFlow. This includes logging model accuracy, infrastructure utilization, configuration details, and code changes.

- **MLFlow Setting Process:**  

  Our group will follow the process below to track our ViT model training with DDP or FSDP:

    1. **Setting Up the Tracking Environment and Object Storage:**  
        We will configure the environment for MLFlow by:
        - Setting the tracking URI via an environment variable to point to our remote MLFlow tracking server.
        - Defining an experiment name so that all logs, metrics, and model artifacts are associated with this experiment.
        - Configuring MinIO as the object storage backend for MLFlow. 
  
    2. **Integrating MLFlow Logging into the Training Script:**  
        In our revised training script, we will:
        - Initialize an MLFlow run at the beginning of the training process.
        - Use MLFlow’s automatic logging for PyTorch to capture details such as model architecture, optimizer settings, and hyperparameters.
        - Log key metrics (e.g., loss, accuracy) during training.
        - Save the final model as an artifact. With MinIO configured as the artifact store, these outputs will be automatically saved there.
  
    3. **DDP and FSDP Training:**  
        - We will import and integrate MLFlow logging code specific to PyTorch.
        - Run the PyTorch training code with MLFlow logging enabled.
        - Optionally, use MLFlow autolog features in combination with PyTorch Lightning for enhanced logging in distributed settings.
  
    4. **Logging Training Metrics and Verification:**  
        - At the end of each training epoch, our revised training strategy will log key metrics (e.g., average training loss, test loss, and accuracy).
        - Finally, we will verify that all experiment details, system metrics, and model artifacts are correctly tracked by using the MLFlow UI.

Distributed Training & Job Scheduling

- **Open A Ray cluster:**
  
  Our group follows the lab process to set up and manage the Ray cluster:
  
    1. **Build a Container Image for Ray Worker Nodes:**  
         - Create a container image with Ray and ROCm installed.
    
    2. **Bring Up the Ray Cluster with Docker Compose:**  
         - Launch the Ray head node and worker nodes using a Docker Compose file.
         - The cluster will include one head node (for scheduling and managing jobs) and two worker nodes.
       
    3. **Start a Jupyter Notebook Container (Without GPUs):**  
         - Run a Jupyter container that is used solely to submit jobs to the Ray cluster.
    
    4. **Access the Ray Dashboard:**  
         - The Ray head node serves a dashboard on port `8265`.  
         - In a browser, open the dashboard using your public IP address (e.g., `http://<YOUR_IP>:8265`).
    
- **Ray Train Process:**  

  Our process for building and managing training and hands-on jobs using Ray Train is as follows:
  
    1. **Preparation:**  
        - Package all python code (VIT), configurations, and environment files (such as `requirements.txt` and `runtime.json`) into my working directory.
    
    2. **Runtime Environment Setup:**  
        - Create a `runtime.json` that specifies:
          - The list of required Python packages (via `requirements.txt`).
          - Build Essential environment variables.
    
    3. **Submitting a Job to the Ray Cluster:**  
        - Use the `ray job submit` command from your Jupyter container to dispatch your training job to the Ray cluster.
        - Specify the required resources (e.g., number of GPUs and CPUs) for the job. For example:
          ```bash
          ray job submit --runtime-env runtime.json --entrypoint-num-gpus 1 --entrypoint-num-cpus 8 --verbose --working-dir . -- python your_training_script.py
          ```
        - This command packages your current working directory, applies the runtime environment, and directs Ray to run your training job on the available worker nodes.
    
    4. **Monitoring and Verification:**  
        - Access the Ray dashboard to verify that the cluster is running properly.
        - Check that the head node and all worker nodes are online and that resource allocations (GPUs/CPUs) match your job requirements.
        - Inspect job logs to ensure that training is proceeding as expected.
    
    5. **Using Ray Train:**  
        - Integrate Ray Train within my training script to manage distributed training, including handling checkpoints and recovering from interruptions.

### Model serving and monitoring platforms

<!-- Make sure to clarify how you will satisfy the Unit 6 and Unit 7 requirements, 
and which optional "difficulty" points you are attempting. -->

### Data pipeline

<!-- Make sure to clarify how you will satisfy the Unit 8 requirements,  and which 
optional "difficulty" points you are attempting. -->

### Continuous X

<!-- Make sure to clarify how you will satisfy the Unit 3 requirements,  and which 
optional "difficulty" points you are attempting. -->
