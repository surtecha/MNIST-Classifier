# MNIST Classifier

<p align="center">
  <img width="251" height="248" src="https://user-images.githubusercontent.com/91011302/225112209-f08013fc-ee68-459e-b2ec-83895bfa7e47.png">
</p>

This project aims to classify handwritten digits using two different approaches: **_K-Nearest Neighbors (kNN)_** algorithm and **_Dense Neural Networks_**. The dataset utilized for this task is the MNIST database, which consists of 28x28 pixel grayscale images of handwritten digits.

## K-Nearest Neighbors (kNN) Algorithm

K-Nearest Neighbors is a simple and effective machine learning algorithm used for classification tasks. The basic idea behind kNN is to classify a data point based on the majority class of its k nearest neighbors in the feature space. In this project, we use the Euclidean distance metric to measure the similarity between data points.

### Algorithm Overview:

1. **Training:** The algorithm stores all the training data points and their corresponding labels.
2. **Prediction:** When classifying a new data point, kNN calculates the distances between the new point and all the points in the training set.
3. **Voting:** It selects the k nearest data points (neighbors) based on the calculated distances.
4. **Classification:** The majority class among the selected neighbors is assigned to the new data point.

### Formula for Euclidean Distance:

It seems like the LaTeX formatting for the formula didn't render properly. Let me correct that for you:

The formula for Euclidean distance is calculated as:

$d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$
    
Where:
- $(x_1$, $y_1)$ are coordinates of the first point.
- $(x_2$, $y_2)$ are coordinates of the second point.
- $d$ is the Euclidean distance between $(x_1$, $y_1)$ and $(x_2$, $y_2)$
This formula represents the calculation of the Euclidean distance, which is a crucial step in the kNN algorithm for measuring the similarity between data points.

## Dense Neural Networks

Neural networks, particularly dense neural networks, have been widely used for classification tasks due to their ability to learn complex patterns in data. In this project, a dense neural network with two hidden layers is employed to classify handwritten digits.

### Architecture:

The neural network architecture consists of an input layer, two hidden layers, and an output layer. The input layer has neurons corresponding to the flattened 28x28 pixel input images. The two hidden layers utilize Rectified Linear Unit (ReLU) and Sigmoid activation functions, respectively, for introducing non-linearity into the model. The output layer has neurons representing the classes (digits 0-9) and employs softmax activation for multi-class classification.

### Training:

The network is trained using backpropagation and gradient descent optimization techniques. During training, the weights of the network are adjusted iteratively to minimize the classification error using a specified loss function (e.g., categorical cross-entropy).

### Performance:

Through experimentation and optimization, the neural network model achieves an accuracy of 97% on the MNIST dataset. This accuracy represents the ability of the model to correctly classify handwritten digits.

These two approaches, kNN and dense neural networks, demonstrate different strategies for handwritten digit classification, each with its own strengths and limitations. Experimentation with various algorithms and architectures can further enhance the accuracy and robustness of the classifier.
