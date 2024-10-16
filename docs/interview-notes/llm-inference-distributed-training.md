## PyTorch and Distributed Training Basics

### What is `torch.distributed` and how does it enable distributed training in PyTorch? How does PyTorch’s `torchrun` utility work?

The `torch.distributed` package in PyTorch is designed to facilitate distributed training across multiple nodes and GPUs. It provides a set of communication primitives and tools to enable efficient parallelism.
- **Multiprocess Parallelism**: torch.distributed allows for synchronous training across multiple processes, which can run on one or more machines. Each process can manage its own model parameters and optimizers, which are synchronized during training.
- **Communication Primitives**: It includes various collective operations such as all-reduce, broadcast, and gather, which are essential for synchronizing gradients and model parameters among different processes.
- **Backend Support**: The package supports different backends like NCCL (for GPU communication), Gloo (for CPU and GPU), and MPI, providing flexibility based on the hardware configuration.

`torchrun` is a utility introduced to simplify launching distributed training jobs in PyTorch. 
- **Fault Tolerance**: torchrun includes built-in mechanisms to handle worker failures gracefully. If a worker fails, it can automatically restart the processes from the last saved snapshot of the training job.
- **Elasticity**: It supports dynamic scaling of resources, allowing the number of nodes to change between minimum and maximum sizes during job execution. 
```sh
torchrun --nnodes=2 --nproc_per_node=4 --rdzv-backend=c10d --rdzv-endpoint=<host>:<port> your_training_script.py
```
**rdzv** refers to **rendezvous**, which is a mechanism used to manage the initialization and coordination mechanism that allows multiple distributed processes to find each other and establish a communication framework across different nodes.
rdzv-backend, specifies the backend used for the rendezvous process. In this case, c10d: A backend that uses a strongly consistent key-value store for managing worker groups. etcd is another alternative.

### Rank vs. World Size vs. Local Rank

- **Rank**: An unique identifier assigned to each process within a distributed training setup. It is an integer that ranges from '0' to 'world_size - 1'.
- **World Size**: Refers to the total number of processes participating in the distributed training job. It is essentially the size of the entire group of processes.
- **Local Rank**: An identifier for a process within its local node (machine). It is particularly useful when multiple processes are running on the same machine, each using a different GPU. Local rank helps assign specific GPU devices to each process.

### Explain the different backends available in torch.distributed. When would you use NCCL versus Gloo?

The choice largely depends on the **type of tensors** communication used (CPU vs. GPU) to decide between NCCL (NVIDIA Collective Communications Library) and Gloo for distributed training in PyTorch.

NCCL library is specifically designed for high-performance inter-GPU communication. It implements collective communication primitives (like all-reduce, broadcast, etc.) that allow GPUs to efficiently share data during training.

|Feature/Use Case|NCCL|Gloo|
|---|---|---|
|Tensor Type|GPU Tensors|CPU Tensors|
|Performance on GPUs|High, optimized for NVIDIA hardware|Lower compared to NCCL|
|Multi-Node Support|Excellent with InfiniBand|Good but less optimized than NCCL|
|Fallback Option|Not applicable|Recommended if NCCL fails|
|Network Type|Best with high-speed interconnects|Suitable for Ethernet|
|Mixed Environments Support|Yes (GPU only)|Yes (CPU + GPU)|

### How does distributed data parallel (DDP) differ from data parallel (DP) in PyTorch?

## Fully Sharded Data Parallel (FSDP)

### What is Fully Sharded Data Parallel (FSDP) and how does it differ from standard DDP?

In DDP, each process maintains a complete replica of the model's parameters, gradients, and optimizer states. This means that every GPU has its own copy of the entire model.

The backward pass, DDP uses `all-reduce` operations to synchronize gradients across all replicas.

FSDP addresses memory limitations by sharding model parameters, gradients, and optimizer states across multiple GPUs. Each process only holds a portion of the model, significantly reducing memory consumption on each GPU.

Communication Overhead: While FSDP reduces the overall memory footprint, it incurs additional communication costs due to the need for operations like all_gather and reduce_scatter during training. This can lead to increased communication volume but is optimized through techniques like overlapping computation and communication.

In forward path

    Run all_gather to collect all shards from all ranks to recover the full parameter in this FSDP unit

    Run forward computation

    Discard parameter shards it has just collected

In backward path

    Run all_gather to collect all shards from all ranks to recover the full parameter in this FSDP unit

    Run backward computation

    Run reduce_scatter to sync gradients

    Discard parameters.

FSDP’s sharding is to decompose the DDP gradient all-reduce into reduce-scatter and all-gather.

<img src="https://pytorch.org/tutorials/_images/fsdp_sharding.png" />

### How does FSDP manage memory usage compared to DDP, and when would you use FSDP over DDP?

### Explain how FSDP handles parameter sharding and gradient sharding during training.

### What are some challenges related to gradient accumulation when using FSDP?
### How do you ensure efficient model checkpointing and loading in FSDP?

## Model Architecture and Scaling

    How would you modify a model to efficiently scale across multiple nodes using FSDP?
    What is tensor sharding and how is it handled in FSDP during forward and backward passes?
    Describe how to handle batch normalization and other stateful layers in a distributed setting.
    How would you profile and debug performance bottlenecks when training large models with FSDP across multiple GPUs?
    Explain how to fine-tune a large pre-trained model using FSDP while avoiding out-of-memory issues.

## Cluster and GPU Management

    How does PyTorch handle device placement across multiple GPUs, and what role do torch.cuda and torch.device play?
    What are the key considerations for setting up an optimal GPU cluster for training large deep learning models?
    How do you handle GPU memory fragmentation issues in large-scale distributed training?
    What are some techniques for improving communication efficiency between GPUs and nodes in a distributed setup?

## Error Handling, Debugging, and Optimization

### How do you handle node failures during distributed training, and what strategies are there for fault tolerance?

A robust checkpointing process followed to allow allow recovery from failures with minimal data loss by saving training state periodically. The training state saved is:
- **Model parameters**: Weights and biases of the neural network.
- **Optimizer states**: Information about the optimizer (e.g., momentum values).
- **Training state**: Current epoch, iteration number, and any other relevant metrics.

PyTorch's Distributed Checkpointing (DCP) library is used tom
- Save model parameters and optimizer states in a distributed manner across all nodes.
- Ensuring that each rank saves its part of the model state, which allows for parallel saving and loading.
- Typically only one rank (e.g., rank 0) is responsible for writing the checkpoint to persistent storage to avoid race conditions. Other ranks can synchronize using barriers to ensure all processes are aligned.

 Saving checkpoints code snippet in PyTorch,
```python
import torch
import torch.distributed as dist

def save_checkpoint(model, optimizer, epoch, filename):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(checkpoint, filename)

if dist.get_rank() == 0:
    save_checkpoint(model, optimizer, epoch, 'checkpoint.pth')
dist.barrier()  # Ensure all ranks wait until the checkpoint is saved
```

When resuming training after a failure, load the checkpoint into all ranks. This involves:
- Reading the checkpoint file and restoring model parameters and optimizer states.
- Synchronizing all ranks to ensure they start from the same state.

Loading checkpoints code snippet,

```python
def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']
```
