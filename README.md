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

The formula for Euclidean distance is calculated as:

$d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$
    
Where:
- $(x_1$, $y_1)$ are coordinates of the first point.
- $(x_2$, $y_2)$ are coordinates of the second point.
- $d$ is the Euclidean distance between $(x_1$, $y_1)$ and $(x_2$, $y_2)$
This formula represents the calculation of the Euclidean distance, which is a crucial step in the kNN algorithm for measuring the similarity between data points.

## Dense Neural Networks

Neural networks, particularly dense neural networks, have been widely used for classification tasks due to their ability to learn complex patterns in data. In this project, a dense neural network with two hidden layers is employed to classify handwritten digits.

### Model Architecture:

The neural network model consists of the following layers:

1. **Input Layer:**
   - Type: Flatten
   - Input Shape: (28, 28)
   - Description: This layer flattens the 28x28 pixel input images into a 1D array.

2. **Hidden Layer:**
   - Type: Dense
   - Number of Neurons: 100
   - Activation Function: ReLU (Rectified Linear Unit)
   - Formula: $(f(x) = max(0, x))$
   - Description: This layer applies the rectified linear activation function to introduce non-linearity to the model.

3. **Output Layer:**
   - Type: Dense
   - Number of Neurons: 10
   - Activation Function: Sigmoid
   - Formula: $`\sigma(x) = \frac{1}{1 + e^{-x}}`$
   - Description: This layer applies the sigmoid activation function to produce class probabilities for the 10 possible digits (0-9).

### Model Compilation

Before training the model, it needs to be compiled with the appropriate optimizer, loss function, and evaluation metrics.

- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Crossentropy
- **Metrics:** Accuracy

### Model Training

To train the model, use the `fit` method with the training data.

```python
model.fit(X_train, y_train, epochs=10)
