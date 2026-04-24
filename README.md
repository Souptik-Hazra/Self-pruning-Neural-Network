# Self-Pruning Neural Network (SPNN)

This repository contains an implementation of a Self-Pruning Neural Network (SPNN) using PyTorch. The project explores a dynamic pruning strategy where the network learns to identify and remove redundant connections during the training process, rather than as a post-training optimization step.

## Overview

The core of this project is the `PrunableLinear` layer, a custom implementation of a fully connected layer that incorporates learnable "gate scores" for each weight. These scores are passed through a sigmoid function to generate gates ($g \in [0, 1]$) that modulate the weights during the forward pass.

To encourage network sparsity, an $L_1$ regularization term is applied to the sigmoid gate values. This creates a constant gradient pressure that drives non-essential gate values toward zero, effectively "pruning" the corresponding connections without significantly impacting model accuracy.

## Key Features

- **Custom Prunable Layers**: Implementation of `PrunableLinear` which manages both weights and gate parameters.
- **Dynamic Sparsity**: Sparsity is optimized concurrently with cross-entropy loss, allowing the model to adapt its architecture to the data.
- **CIFAR-10 Benchmarking**: The architecture is evaluated on the CIFAR-10 dataset to demonstrate the trade-off between model complexity and performance.
- **Visualization Tools**: Includes scripts for analyzing gate value distributions and tracking sparsity levels throughout the training duration.

## Technical Implementation

### Pruning Mechanism
The weight modulation is defined as:
$$W_{effective} = W \odot \sigma(S)$$
where $W$ represents the weight matrix, $S$ the gate scores, and $\odot$ denotes element-wise multiplication.

The total loss function used for training is:
$$\mathcal{L} = \mathcal{L}_{CE} + \lambda \sum \sigma(S)$$
where $\lambda$ controls the intensity of the sparsity constraint.

### Project Structure
- `self_pruning_nn.ipynb`: The primary Jupyter Notebook containing the model architecture, training pipeline, and experimental results.
- `gate_distributions.png`: Visualization of how gates polarize toward 0 or 1 under different regularization strengths.

## Getting Started

### Prerequisites
Ensure you have the following dependencies installed:
- Python 3.8+
- PyTorch
- Torchvision
- NumPy
- Matplotlib

### Usage
Clone the repository and execute the Jupyter Notebook to reproduce the experiments:
```bash
pip install torch torchvision numpy matplotlib
```

## Experimental Results

The model was tested with varying values of the sparsity coefficient ($\lambda$):
- **Weak Regularization ($\lambda = 1e-5$)**: High accuracy is maintained, but negligible pruning occurs.
- **Balanced Regularization ($\lambda = 1e-4$)**: Achieves a significant reduction in active parameters while maintaining competitive accuracy.
- **Aggressive Regularization ($\lambda = 1e-3$)**: High sparsity levels are achieved, though at the cost of a noticeable decline in classification accuracy.

The resulting bimodal distribution of gate values confirms that the network successfully distinguishes between critical and redundant connections.

## Conclusion
This implementation demonstrates that $L_1$ regularization on sigmoid-gated weights is an effective method for end-to-end differentiable pruning. It provides a robust framework for developing more efficient neural architectures.