## PyTorch and Distributed Training Basics

    What is torch.distributed and how does it enable distributed training in PyTorch?
    How does PyTorch’s torchrun utility work, and what are its benefits compared to torch.distributed.launch?
### Explain the different backends available in torch.distributed. When would you use NCCL versus Gloo?

The choice largely depends on the **type of tensors** used (CPU vs. GPU) to decide between NCCL (NVIDIA Collective Communications Library) and Gloo for distributed training in PyTorch.

|Feature/Use Case|NCCL|Gloo|
|---|---|---|
|Tensor Type|GPU Tensors|CPU Tensors|
|Performance on GPUs|High, optimized for NVIDIA hardware|Lower compared to NCCL|
|Multi-Node Support|Excellent with InfiniBand|Good but less optimized than NCCL|
|Fallback Option|Not applicable|Recommended if NCCL fails|
|Network Type|Best with high-speed interconnects|Suitable for Ethernet|
|Mixed Environments Support|Yes (GPU only)|Yes (CPU + GPU)|

    How does distributed data parallel (DDP) differ from data parallel (DP) in PyTorch?
    What are the key challenges when training deep learning models on multiple nodes and GPUs?

## Fully Sharded Data Parallel (FSDP)

    What is Fully Sharded Data Parallel (FSDP) and how does it differ from standard DDP?
    How does FSDP manage memory usage compared to DDP, and when would you use FSDP over DDP?
    Explain how FSDP handles parameter sharding and gradient sharding during training.
    What are some challenges related to gradient accumulation when using FSDP?
    How do you ensure efficient model checkpointing and loading in FSDP?

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
