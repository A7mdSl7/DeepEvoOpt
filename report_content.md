# DeepEvoOpt Project Report Content

## 1. Dataset Description
- **Name**: Fashion-MNIST
- **Type**: Image Classification
- **Content**: 60,000 training images, 10,000 test images. 28x28 grayscale.
- **Classes**: 10 (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).
- **Preprocessing**: Normalized to [-1, 1]. Split into Train (90%) and Validation (10%).

## 2. Model Architectures

### Convolutional Neural Network (CNN)
- **Input**: 28x28x1
- **Layer 1**: Conv2d (k=3/5, filters=16/32/64) -> ReLU -> MaxPool(2x2)
- **Layer 2**: Conv2d (k=3/5, filters=32/64/128) -> ReLU -> MaxPool(2x2)
- **Flatten**
- **Layer 3**: Linear (units=64/128/256) -> ReLU -> Dropout (0.1-0.5)
- **Output**: Linear (10 units)

### Multi-Layer Perceptron (MLP)
- **Input**: 784 (flattened)
- **Layer 1**: Linear (units=64-512) -> ReLU -> Dropout
- **Layer 2**: Linear (units=32-128) -> ReLU -> Dropout
- **Output**: Linear (10 units)

## 3. Hyperparameters Optimized
- **Learning Rate**: [0.0001, 0.01]
- **Batch Size**: [32, 64, 128]
- **Dropout Rate**: [0.1, 0.5]
- **Optimizer**: Adam, SGD
- **Model Specific**: Kernel sizes, Number of filters, Hidden units.

## 4. Optimizers Overview
- **GA**: Standard Genetic Algorithm with tournament selection, uniform crossover, and mutation.
- **PSO**: Particle Swarm Optimization with inertia and cognitive/social components.
- **GWO**: Grey Wolf Optimizer mimicking leadership hierarchy (alpha, beta, delta).
- **ACO**: Continuous Ant Colony Optimization using Gaussian sampling around guide solutions.
- **Firefly**: Firefly Algorithm based on attractiveness and light intensity.
- **ABC**: Artificial Bee Colony with employed, onlooker, and scout bees.
- **OBC-WOA**: Whale Optimization Algorithm enhanced with Opposition-Based Learning and Chaotic maps.
- **FCR**: Fitness-Centered Recombination focusing on recombining elite solutions.
- **FCGWO**: GWO with fuzzy-logic-based parameter adaptation for exploration/exploitation balance.

## 5. Experimental Setup
- **Population Size**: 10 (default)
- **Iterations**: 5-10 (short runs for demo), can be increased.
- **Objective**: Minimize Validation Loss on Fashion-MNIST (3-5 epochs per evaluation).
- **Hardware**: GPU (CUDA) if available, else CPU.

## 6. Results
*(To be filled after running experiments)*
- **Best Optimizer**: [Insert Best Optimizer]
- **Best Accuracy**: [Insert Best Accuracy]
- **Convergence**: See generated plots in `results/figures`.
