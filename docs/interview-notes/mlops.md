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
---

- [Building and Deploying Machine Learning Pipelines](https://www.datacamp.com/tutorial/kubeflow-tutorial-building-and-deploying-machine-learning-pipelines)

- [7 Frameworks for Serving LLMs - a comprehensive guide into LLMs inference and serving with detailed comparison](https://betterprogramming.pub/frameworks-for-serving-llms-60b7f7b23407)
- [Case Study: Amazon Ads Uses PyTorch and AWS Inferentia to Scale Models for Ads Processing](https://pytorch.org/blog/amazon-ads-case-study/)
- [GPU - CUDA Programming Model](https://luniak.io/cuda-neural-network-implementation-part-1/)
- [GPU - CUDA Interview questions on CUDA Programming?](https://stackoverflow.com/questions/1958320/interview-questions-on-cuda-programming)
- [Improving GPU Utilization in Kubernetes](https://developer.nvidia.com/blog/improving-gpu-utilization-in-kubernetes/)
- [Kubeflow pipelines (part 1) — lightweight components](https://medium.com/@gkkarobia/kubeflow-pipelines-part-1-lightweight-components-a4a3c8cb3f2d)
- [Automate prompt tuning for large language models using KubeFlow Pipelines](https://developer.ibm.com/tutorials/awb-automate-prompt-tuning-for-large-language-models/)
- [Building a ML Pipeline from Scratch with Kubeflow – MLOps Part 3](https://blogs.cisco.com/developer/machinelearningops03)

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

### Kubeflow Pipelines

Pipelines are used to chain multiple steps into an ML lifecycle - data prep, train, evaluate, deploy.



### What is Container Op in kubeflow pipelines? Isn't it too heavy creating containers for every step?

A ContainerOp represents an execution of a Docker container as a step in the pipeline. Some key points:
- Every step in the pipeline is wrapped as a ContainerOp. This encapsulates the environment and dependencies for that step.
- Behind the scenes, the ContainerOp creates a Kubernetes Pod with the specified Docker image to run that step.
- So yes, containerizing every step introduces computational overhead vs just running Python functions. The containers can be slow to spin up and add resource usage.
- However, the benefit is that each step runs in an isolated, reproducible environment. This avoids dependency conflicts between steps.
- ContainerOps also simplify deployment. The pipeline itself is a portable Kubernetes spec that can run on any cluster.

For Python code, the func_to_container_op decorator allows you to convert Python functions to ContainerOps easily.

For performance critical sections, you can optimize by:
- Building custom slim Docker images
- Using the **dsl.ResourceOp** construct for non-container steps
- Group multiple steps in one ContainerOp
-  Tuning Kubernetes Pod resources

So in summary, ContainerOps trade off some performance for better encapsulation, portability and reproducibility. For a production pipeline, optimizing performance is still important.

ResourceOps in Kubeflow Pipelines allow defining pipeline steps that don't execute as Kubernetes Pods or ContainerOps.

### Kubeflow Pipelines Vs Jobs

- **Jobs** are standalone, reusable ML workflow steps e.g. data preprocessing, model training.
- **Pipelines** stitch together jobs into end-to-end ML workflows with dependencies.

- Use jobs to encapsulate reusable ML functions - data processing, feature engineering etc.
- Use jobs within a pipeline to break up the pipeline into reusable components.
- Kubeflow jobs leverage composable kustomize packages making reuse easier. Kubernetes jobs use vanilla pod templates.

## Model Serving

### Difference between saving ML model & packaging a model? What are differtent formats for packaging model?

**Saving a model** involves persisting just the core model artifacts like weights, hyperparameters, model architecture etc. This captures the bare minimum needed to load and use the model later.

**Packaging a model** is more comprehensive - it bundles the model with any additional assets needed to deploy and serve the model. This allows portability. Model packaging includes:
- Saved model files
- Code dependencies like Python packages
- Inference code
- Environment specs like Dockerfile
- Documentation & Metadata like licenses, model cards etc

Containerizing can be considered a form of model packaging, as it encapsulates the model and dependencies in a standardized unit for deployment.

Model packaging formats:

- Docker image - Bundle model as microservice
-  Model archival formats - ONNX, PMML
- Python package - For use in Python apps

> [Automating Kubeflow Pipelines with GitOps, GitHub Actions?](https://www.weave.works/blog/automating-kubeflow-pipelines-with-gitops-github-actions-and-weave-flagger)

Why not use containers as the delivery medium for models?

It is fortunate that Kubernetes supports the concept of init-containers inside a Pod. When we serve different machine learning models, we change only the model, not the serving program. This means that we can pack our models into simple model containers, and run them as the init-container of the serving program. We also include a simple copy script inside of our model containers that copies the model file from the init container into a volume so that it can be served by the model serving program.

https://github.com/chanwit/wks-firekube-mlops/blob/master/example/model-serve.yaml


### Kubeflow operators

Here is a list of some of the key operators in Kubeflow:

Kubeflow Pipelines operators:

    Argoproj Workflow Operator - Manages and executes pipelines
    Persistent Agent - Caches data for pipelines agents
    ScheduledWorkflow Operator - Handles scheduled/cron workflows

Notebooks operators:

    Jupyter Web App Operator - Manages Jupyter notebook web interfaces
    Notebook Controller Operator - Handles lifecycle of notebook servers

Serving operators:

    KFServing Operator - Manages model inference services
    Seldon Deployment Operator - Manages running Seldon inference graphs
    Inference Services Operator - Deprecated operator for TF Serving

Training operators:

    MPI Operator - Handles distributed MPI training jobs
    PyTorch Operator - Manages PyTorch training jobs
    TFJob Operator - Manages TensorFlow training jobs

Misc operators:

    Kubeflow Namespace Operator - Installs components in kubeflow namespace
    Profile Controller - Records detailed profiles of runs
    Studyjob Operator - Hyperparameter tuning using Katib
    Volumes Web App - Manages PVC and volumes

The core operators for pipelines, notebooks and model serving/inference provide end-to-end ML workflow functionality on Kubernetes. The training operators support orchestrated model training jobs.

There are also various utility operators for management, tuning, monitoring etc. Many operators are being consolidated under KServe umbrella.

### Hyperparameter tuning

> Write code for hyperparameter tuning for xgboost using kubeflow pipelines? where the multi fold cross validation information is stored? how best model is detected?

```python
import kfp
from kfp import dsl
from kfp.components import func_to_container_op

def xgboost_train(params):
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model

@func_to_container_op
def log_metrics(model):
    cv_scores = cross_validate(model, X_train, y_train, cv=5)
    print(cv_scores)
    return model

@dsl.pipeline(
    name='XGBoost-tuning',
    description='Tuning XGBoost hyperparameters using Katib'
)
def xgboost_tuning_pipeline(
    learning_rate = hp.choice(0.01, 0.1),
    n_estimators = hp.choice(100, 200)    
):

    xgb_train_op = dsl.components.func_to_container_op(xgboost_train)
    best_hp = tuner.search(xgb_train_op, {
        "learning_rate": learning_rate,
        "n_estimators": n_estimators
    })

    final_model = xgb_train_op(best_hp.to_dict())
    log_metrics(final_model)

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(xgboost_tuning_pipeline, __file__ + '.yaml')
```

### KServe 

- KServe is a Kubernetes operator for serving, managing and monitoring machine learning models on Kubernetes.
- KServe supports multiple inference runtimes like TensorFlow, PyTorch, ONNX, XGBoost.
- KServe can leverage GPU acceleration libraries like TensorRT for optimized performance.
- For PyTorch, KServe integrates with TorchServe to serve PyTorch models.

### TorchScript

- Torchscript is a way to create serializable and optimizable models from PyTorch code.
- It is a Python scripting language that can be used as an intermediate representation (IR) for PyTorch models
- TorchScript is used to optimize and serialize PyTorch models for deployment to production environments


### TensorRT & Triton Inference Server

TensorRT optimizes and executes compatible subgraphs, letting deep learning frameworks execute the remaining graph

- TensorRT is designed to optimize deep learning models for inference on NVIDIA GPUs, which results in faster inference times.
- TensorRT performs various optimizations on the model graph, including layer fusion, precision calibration, and dynamic tensor memory management to enhance inference performance.

During the optimization process, TensorRT performs the following steps:

- Parsing: TensorRT parses the trained model to create an internal representation.
- Layer fusion: TensorRT fuses layers in the model to reduce memory bandwidth requirements and increase throughput.
- Precision calibration: TensorRT selects the appropriate precision for each layer in the model to maximize performance while maintaining accuracy.
- Memory optimization: TensorRT optimizes the memory layout of the model to reduce memory bandwidth requirements and increase throughput.
- Kernel selection: TensorRT selects the best kernel implementation for each layer in the model to maximize performance.
- Dynamic tensor memory management: TensorRT manages the memory required for intermediate tensors during inference to minimize memory usage.

TensorRT is better than TorchScript in terms of performance.

[TensorRT Optimization Guide](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-723/pdf/TensorRT-Best-Practices.pdf)
