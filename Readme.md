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
## Installation

Ensure you have Python installed along with all the Required Dependencies. <br>You can install dependencies using:
```bash
pip install -r requirements.txt
```
<br>

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/sawaipratap/Neural-Network-Implementation-using-NumPy.git 
   cd Neural-Network-Implementation-using-NumPy
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

