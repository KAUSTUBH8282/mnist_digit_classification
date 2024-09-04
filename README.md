# MNIST Handwritten Digit Classification using PyTorch

This project is a simple implementation of a neural network to classify handwritten digits from the MNIST dataset. It is written in Python using PyTorch and is designed to be run on Google Colab.

## Project Overview

- The model is a simple feedforward neural network with three layers.
- It uses the MNIST dataset, which consists of 28x28 grayscale images of digits (0-9).
- The network is trained to classify these images into one of the 10 digit classes.

## Colab Notebook

You can run this project on Google Colab without setting up a local environment.

[Run on Google Colab](https://colab.research.google.com/)

## Steps to Run on Colab

1. Open the Colab notebook [here](https://colab.research.google.com/).
2. Upload the project code to the Colab environment.
3. Install necessary dependencies if not pre-installed:
    ```python
    !pip install torch torchvision
    ```
4. Run the cells to:
    - Load and preprocess the MNIST dataset.
    - Define the neural network architecture.
    - Train the network for 5 epochs.
    - Evaluate the accuracy on the test dataset.

## Neural Network Architecture

The neural network consists of:
- **Input Layer**: Flattens the 28x28 image into a vector of size 784.
- **Hidden Layers**:
    - First hidden layer with 128 neurons and ReLU activation.
    - Second hidden layer with 64 neurons and ReLU activation.
- **Output Layer**: A fully connected layer with 10 output units (for the 10 digit classes).

## Results

The model is trained for 5 epochs and achieves an accuracy of approximately 97% on the test dataset.

## Dataset

- The dataset used is the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), which contains 60,000 training images and 10,000 testing images of handwritten digits.
- The dataset is loaded using `torchvision.datasets`.

## Requirements

No local setup is needed if you are running on Google Colab. The required libraries are pre-installed, but you can manually install them using the following command:

```bash
pip install torch torchvision
