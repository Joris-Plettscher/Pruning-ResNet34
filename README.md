# Pruning-ResNet34

## Project Overview

This project demonstrates the pruning of a ResNet34 model on the CIFAR-10 dataset using the Torch-Pruning library. The goal was to reduce the model size while maintaining high accuracy.

## Repository Structure

- **PruningResNet34.ipynb**: The Jupyter notebook containing the code for pruning ResNet34. It includes data loading, model training, pruning, and evaluation steps.
  
- **ResNet34_data.csv**: A CSV file that contains the accuracy results from tests conducted during the project.

- **models/**: A directory containing models at different stages of pruning and retraining:
  - **Pre_Retraining/**: Contains models that were pruned but not retrained.
  - **Post_Retraining/**: Contains models that were retrained for 5 epochs after pruning.
  - **Double_Retraining/**: Contains models that were retrained for 10 epochs after pruning.
  - **ResNet34_CIFAR10.pth**: The standard, non-pruned ResNet34 model trained on CIFAR-10.

- **Accuracies.png**: A visualization of the accuracies showing the effectiveness of retraining the models with twice the number of epochs.

## Pruning and Retraining Details

- **Pruning Technique**: The pruning was performed using the Torch-Pruning library, which allows for structured pruning of neural networks.

## Results

The results indicate that retraining the pruned models, especially with a doubled number of epochs, helps in maintaining high accuracy.

![Accuracies.png](https://github.com/Joris-Plettscher/Pruning-ResNet34/blob/main/Accuracies.png?raw=true)

## Usage

To reproduce the results or experiment further:

1. Open the `PruningResNet34.ipynb` notebook.
2. Follow the instructions in the notebook to load data, prune the model, and evaluate performance.
3. Use the models in the `models/` directory as needed for comparison or further analysis.

## Dependencies

- Python 3.x
- PyTorch
- Torchvision
- Torch-Pruning
- Jupyter Notebook
- Matplotlib
- tqdm
- NumPy
- Pandas
