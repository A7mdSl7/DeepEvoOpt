# DeepEvoOpt

DeepEvoOpt is a hyperparameter optimization system for deep learning models (CNN and MLP) on the Fashion-MNIST dataset. It utilizes 9 meta-heuristic optimization algorithms to find the best hyperparameters.

## Project Structure

```
DeepEvoOpt/
├── data/               # Dataset directory
├── notebooks/          # Jupyter notebooks for experiments
├── results/            # Results and logs
├── src/                # Source code
│   ├── models/         # CNN and MLP models
│   ├── optimizers/     # Optimization algorithms (GA, PSO, GWO, etc.)
│   ├── utils/          # Utility functions (data loader, objective fn)
│   ├── train.py        # Final model training script
│   └── run_experiments.py # Main experiment runner
├── tests/              # Tests
├── requirements.txt    # Dependencies
└── report.docx         # Project report
```

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running Experiments via CLI

You can run experiments using `src/run_experiments.py`:

```bash
# Run GA on CNN
python src/run_experiments.py --optimizer ga --model_type cnn --pop_size 10 --max_iter 5

# Run all optimizers
python src/run_experiments.py --optimizer all --model_type cnn
```

### Running Experiments via Notebook

Open `notebooks/Experiments.ipynb` to run experiments interactively and visualize results.

## Implemented Optimizers

1. **GA**: Genetic Algorithm
2. **PSO**: Particle Swarm Optimization
3. **GWO**: Grey Wolf Optimizer
4. **ACO**: Ant Colony Optimization
5. **Firefly**: Firefly Algorithm
6. **ABC**: Artificial Bee Colony
7. **OBC-WOA**: Opposition-based Chaotic Whale Optimization Algorithm
8. **FCR**: Fitness-Centered Recombination
9. **FCGWO**: Fuzzy-Controlled GWO

## Models

- **CNN**: 2 Convolutional layers, MaxPool, Dropout, FC layers.
- **MLP**: 2-3 Fully Connected layers, Dropout.

## Dataset

Fashion-MNIST: 28x28 grayscale images of 10 fashion categories.