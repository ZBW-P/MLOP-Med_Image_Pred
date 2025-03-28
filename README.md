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
| Team member 2                   |                 |                                    |
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

| Name of Data/Model                              | How it was Created                                                                                                                                                                                                                                                                                                                                                                                                                 | Conditions of Use                                                                                                                                                                                                                                                       |
|-------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Pneumonia Detection using CNN (92.6% Accuracy)**  | Retrospectively collected chest X-ray images from pediatric patients (ages 1–5) at Guangzhou Women and Children’s Medical Center. Images were quality-screened and diagnoses confirmed by two expert physicians (with a third for verification) before being split into training, validation, and test sets. For more details, see the [Kaggle Notebook](https://www.kaggle.com/code/madz2000/pneumonia-detection-using-cnn-92-6-accuracy/notebook#Description-of-the-Pneumonia-Datasetand). | For academic and non-commercial research only. Users must properly cite the original dataset source and associated publications. |
| **COVID-19 Radiography Database**                   | Developed by a collaborative team from Qatar University, University of Dhaka, and partner institutions. The dataset aggregates chest X-ray images for COVID-19, Normal, and Viral Pneumonia cases. The initial release contained 219 COVID-19, 1341 Normal, and 1345 Viral Pneumonia images, with subsequent updates increasing COVID-19 cases to 3616 and adding corresponding lung masks. See the [Kaggle Dataset](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/data) for more information. | Released for academic/non-commercial use. Users must cite Chowdhury et al. (IEEE Access, 2020) and Rahman et al. (2020) and adhere to the usage guidelines provided in the dataset metadata. |



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

#### Model training and training platforms

<!-- Make sure to clarify how you will satisfy the Unit 4 and Unit 5 requirements, 
and which optional "difficulty" points you are attempting. -->

#### Model serving and monitoring platforms

<!-- Make sure to clarify how you will satisfy the Unit 6 and Unit 7 requirements, 
and which optional "difficulty" points you are attempting. -->

#### Data pipeline

<!-- Make sure to clarify how you will satisfy the Unit 8 requirements,  and which 
optional "difficulty" points you are attempting. -->

#### Continuous X

<!-- Make sure to clarify how you will satisfy the Unit 3 requirements,  and which 
optional "difficulty" points you are attempting. -->
