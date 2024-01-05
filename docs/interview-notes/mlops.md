# MLOps & Kubeflow

**Table of Contents**
- Overview
- ML Pipeline Tasks
- Kubeflow
  - Overview & Components  
  - Jupyter Notebooks Hub
  - Kubeflow Pipelines
  - Kale Jupyter Notbook to Pipeline Conversion
  - Parameter Tuning using Katib
  - Model Registry
  - Kubeflow Serving
  - Kubeflow Operators
- Kuebflow GPUs

[ml-ops.org](https://ml-ops.org/content/end-to-end-ml-workflow)

<img src="https://ml-ops.org/img/ml-engineering.jpg" width="70%" height="70%" />

<img src="https://www.freecodecamp.org/news/content/images/size/w2000/2021/03/mlops_thumb.png" width="70%" height="70%" />

<img src="https://learn.microsoft.com/en-us/azure/architecture/data-science-process/media/lifecycle/tdsp-lifecycle2.png" width="70%" height="70%" />

### Where is your models stored? What is a model registry (Or) model repository? Is your models stored in S3? What are popular tools for model store/registry? How models are versioned?

### Tell me about your “data preparation & processing” pipelines? Explain about your “Data Acquisition” pipelines?
- Data Acquistion
  - Crawler & Scraper
  - Data Ingestion Processing: Streaming, Batch
- Data Validation
- Data Wrangling & Cleaning
  - Text
    - Raw Text, Tokenization, Word embeddings, 
- Data Versioning

- Data Store?
  - Structured & Unstructured
  - Vector DB vs. PySpark vs. ~MongoDB~

### Provide insights about your "feature engineering" pipelines?

- Exploratory Data Analysis
- Transform, Binning Temporarl
- Clean, Normalize, Impute Missing, Extract Features, Encode Features, 
- Feature Selection

- Feature Store

### Explain ML CI pipelines?

### How to package artifacts?

- Model Packaging
  - .pkl
  - ONNX
  - 

### Explain training pipelines? What experiment pipelines?

- Model Training
- Model validation
- Hyperparameter Tuning
- Best Model Selection
- Cross Validation

### Explain model deployment or inference pipelines (CD- Continuous Delivery)? How production serving is done? How instrumentation works (logging, monitoring, observability)? - TorchServe, TensorRT

- Model Serving
  - KServe Vs. TorchServe
  - K8s, Docker
  - Service - gRPC Vs. REST Vs. Client Libraries

- Performance, Benchmarking
- Scoring & Monitoring

### How A/B testing carried out?

- Experiment Tracking
- 
## Kubeflow

### Overview
Kubeflow enables you to set up CI/CD pipelines for your machine learning workflows. This allows you to automate the testing, validation, and deployment of models, ensuring a smooth and scalable deployment process.
