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

## MLOps Overview
> A picture speaks a thousand words
<table>
    <tr>
        <td><img src="https://ml-ops.org/img/ml-engineering.jpg" /></td>
        <td><img src="https://www.freecodecamp.org/news/content/images/size/w2000/2021/03/mlops_thumb.png" /></td>
        <td><img src="https://learn.microsoft.com/en-us/azure/architecture/data-science-process/media/lifecycle/tdsp-lifecycle2.png" /></td>
    </tr>
</table>

## ML Pipeline Tasks

### What are steps/tasks in “Data Acquisition” “Data preparation & processing” pipelines?
- Data Acquistion
  - Crawler & Scraper
  - Data Ingestion Processing: Streaming, Batch
- Data Validation
- Data Wrangling & Cleaning
  - Text
    - Raw Text, Tokenization, Word embeddings, 
- Data Versioning

### What are steps in "feature engineering" pipelines?

- Exploratory Data Analysis
- Transform, Binning Temporarl
- Clean, Normalize, Impute Missing, Extract Features, Encode Features, 
- Feature Selection
- Feature Store

### How do you package model artifacts?

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


### Where models are stored in Kubeflow? What is a model registry (Or) model repository? Is your models stored in S3? What are popular tools for model store/registry? How models are versioned?

Here are some key points about model storage and registries in Kubeflow:

    Kubeflow provides a centralized model repository for storing, versioning, and managing ML models. This is called the Kubeflow Model Registry.

### Where Kubeflow stores modeles?

- Kubeflow Model Registry stores the model artifacts (weights, hyperparameters etc) in a storage service like S3 or GCS. The metadata (name, version, path etc) is stored in a database like MySQL, PostgreSQL etc.
- Metadata like model name, version etc are stored in a database.
- Using the model registry, models can be versioned, searched, deployed and integrated into ML pipelines in a centralized way. It provides features like model lineage, model security, governance etc.
- To version models, each model build or iteration can be assigned a unique version number or id. New model versions can be registered and looked up by version. Older versions are retained for record keeping.

> Claude/ChatGPT Prompt: Show me Kubeflow code snippet to store a model into model registry? Also show code for to fetch a model from model repository using kubeflow pipelines SDK?

### What are typical containers behind Kubeflow Model Registry?

- Registry Server (kubeflow-model-registry-server) & Registry UI (kubeflow-model-registry-ui)
- ML Metadata DB: ml-pipeline-ui-db, ml-pipeline-mysql
- Minio Object Storage: minio
- PostgreSQL for ML Metadata: kubeflow-ml-pipeline-db
- Model Inference Servers: kfserving-tensorflow-inference-server, triton-inference-server
- Push Model Utility (kubeflow-push-model), Fetch Model Utility (kubeflow-fetch-model) & Model Converter (kubeflow-model-converter)

The push-model container provides a Python SDK and CLI to upload model artifacts and metadata to the registry.

```python
push_model_op = kfp.components.load_component_from_url('kubeflow-push-model')
```
### How to choose between Kubeflow Model Registry and Hugging Face Model Hub?

- Kubeflow registry for internal model management
- HF Model Hub to leverage public/shared models

Hugging Face Model Hub

-Public repository of a broad range of ML models
-Emphasis on easy sharing and use of models
-Integrates with Hugging Face libraries like Transformers
-Includes preprocessed datasets, notebooks, demos

### Kubeflow Pipelines Vs Jobs

## Model Serving

### Difference between saving ML model & packaging a model? What are differtent formats for packaging model?

**Saving a model** involves persisting just the core model artifacts like weights, hyperparameters, model architecture etc. This captures the bare minimum needed to load and use the model later.

**Packaging a model** is more comprehensive - it bundles the model with any additional assets needed to deploy and serve the model. This allows portability. Model packaging includes:
- Saved model files
- Code dependencies like Python packages
- Inference code
- Environment specs like Dockerfile
- Documentation & Metadata like licenses, model cards etc
- 
Containerizing can be considered a form of model packaging, as it encapsulates the model and dependencies in a standardized unit for deployment.

Model packaging formats:

- Docker image - Bundle model as microservice
-  Model archival formats - ONNX, PMML
- Python package - For use in Python apps
