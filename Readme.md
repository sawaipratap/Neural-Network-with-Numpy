# Neural Network Implementation using NumPy

This repository contains an implementation of a simple feedforward neural network using only NumPy. The network is designed to perform binary classification using the sigmoid activation function.
<br><br><br>
## Neural Network Architecture

The network consists of:
- **Input layer**: Two input features (Weight and Height)
- **One hidden layer** with two neurons using the sigmoid activation function
- **Output layer**: A single neuron using the sigmoid activation function
<br><br>
### Architecture Diagram

![Neural Network Architecture](./Model%20Architecture.png)
<br><br>
Activation Function used: **Sigmoid**
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$
<br>

The network follows the following mathematical equations:

$$ H_1 = x_1 W_1 + x_2 W_2 + b_1 $$
$$ H_2 = x_1 W_3 + x_2 W_4 + b_2 $$

<br>

$$ y_{pred} = \sigma(H_1 W_5 + H_2 W_6 + b_3) $$
$$ y_{pred}= \frac{1}{1 + e^{-(H_1 W_5 + H_2 W_6 + b_3)}} $$

<br><br>

## Backpropagation Algorithm

Backpropagation is a fundamental algorithm for training neural networks, enabling them to adjust weights and biases to minimize prediction errors. This implementation utilizes backpropagation to efficiently compute gradients and update parameters.

### How Backpropagation Works
1.	**Forward Pass:** Compute the output of the network by passing input data through each layer.
2.	**Compute Loss:** Calculate the discrepancy between the predicted output and the actual target using a loss function.
3.	**Backward Pass:** Propagate the error backward through the network to compute gradients of the loss with respect to each weight and bias.
4.	**Update Parameters:** Adjust the weights and biases using the computed gradients, typically with an optimization algorithm like stochastic gradient descent.

### Mathematical Formulation

For each neuron, the weight update rule is defined as:

$$ w_{ij} \leftarrow w_{ij} - \eta \frac{\partial E}{\partial w_{ij}} $$



The partial derivative ￼ represents the gradient of the loss with respect to the weight, indicating the direction and magnitude of the weight adjustment needed to reduce the error.

<br>

## Installation

Ensure you have Python installed along with all the Required Dependencies. <br>You can install dependencies using:
```bash
pip install -r requirements.txt
```
<br>

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/sawaipratap/Neural-Network-with-Numpy 
   cd Neural-Network-with-Numpy
   ```
2. Run the Jupyter Notebook or Python script:
   ```bash
   jupyter notebook NeuralNetwork.ipynb
   ```
<br>

## Features
- Feed Forward Network using matrix multiplication
- Sigmoid activation function
- Basic backpropagation
- Training using stochastic gradient descent

<br>

## Scope of Improvements
- Adding support for multiple layers
- Implementing ReLU and other activation functions
- Support for different loss functions

