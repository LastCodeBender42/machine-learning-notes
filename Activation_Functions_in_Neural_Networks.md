### Activation Functions in Neural Networks

An activation function in a neural network defines the output of a neuron given an input or set of inputs. It introduces non-linearity into the model, enabling it to learn and represent complex patterns in data. Without activation functions, a neural network would behave like a simple linear regression model, regardless of the number of layers.

### Common Activation Functions

1. **Sigmoid Function**:
   \[
   \sigma(x) = \frac{1}{1 + e^{-x}}
   \]
   - **Range**: (0, 1)
   - **Usage**: Often used in binary classification problems.
   - **Example**:
     ```python
     import tensorflow as tf
     x = tf.constant([-1.0, 0.0, 1.0])
     sigmoid = tf.nn.sigmoid(x)
     print(sigmoid.numpy())  # Output: [0.26894142, 0.5, 0.7310586]
     ```

2. **Tanh (Hyperbolic Tangent) Function**:
   \[
   \tanh(x) = \frac{2}{1 + e^{-2x}} - 1
   \]
   - **Range**: (-1, 1)
   - **Usage**: Commonly used in hidden layers of neural networks.
   - **Example**:
     ```python
     import tensorflow as tf
     x = tf.constant([-1.0, 0.0, 1.0])
     tanh = tf.nn.tanh(x)
     print(tanh.numpy())  # Output: [-0.7615942, 0.0, 0.7615942]
     ```

3. **ReLU (Rectified Linear Unit) Function**:
   \[
   \text{ReLU}(x) = \max(0, x)
   \]
   - **Range**: [0, ∞)
   - **Usage**: Widely used in hidden layers of neural networks due to its simplicity and effectiveness.
   - **Example**:
     ```python
     import tensorflow as tf
     x = tf.constant([-1.0, 0.0, 1.0])
     relu = tf.nn.relu(x)
     print(relu.numpy())  # Output: [0.0, 0.0, 1.0]
     ```

4. **Leaky ReLU Function**:
   \[
   \text{Leaky ReLU}(x) = 
   \begin{cases} 
   x & \text{if } x > 0 \\
   \alpha x & \text{if } x \leq 0
   \end{cases}
   \]
   - **Range**: (-∞, ∞)
   - **Usage**: Used to solve the "dying ReLU" problem by allowing a small, non-zero gradient when the unit is not active.
   - **Example**:
     ```python
     import tensorflow as tf
     x = tf.constant([-1.0, 0.0, 1.0])
     leaky_relu = tf.nn.leaky_relu(x, alpha=0.1)
     print(leaky_relu.numpy())  # Output: [-0.1, 0.0, 1.0]
     ```

### Importance of Activation Functions

Activation functions introduce non-linearity into the neural network, which allows it to learn and represent more complex patterns. Without non-linear activation functions, a neural network, regardless of its depth, would be equivalent to a single-layer perceptron, which can only model linear relationships.

Each activation function has its own characteristics and is chosen based on the specific requirements of the model and the nature of the data. For instance, ReLU is popular for deep networks because it helps mitigate the vanishing gradient problem, while sigmoid and tanh are often used in the output layers of binary and multi-class classification problems, respectively.


Let's consider a scenario where you are working with gene expression data to build a neural network model for classifying types of cancer based on the expression levels of various genes.

### Example: Using Activation Functions with Gene Expression Data

#### 1. Preparing the Data
Suppose you have a dataset where each row represents a sample, each column represents a gene, and the target variable indicates the type of cancer. Here’s a simplified example:

- **Features**: Gene expression levels (e.g., `gene1`, `gene2`, ..., `geneN`)
- **Target**: Cancer type (e.g., 0 for type A, 1 for type B)

#### 2. Building the Neural Network
We'll build a neural network model using TensorFlow, where activation functions play a crucial role in transforming the input gene expression data through the network layers.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Example gene expression data (for illustration purposes)
import numpy as np
np.random.seed(42)
num_samples = 100
num_genes = 50
X = np.random.rand(num_samples, num_genes)  # Gene expression levels
y = np.random.randint(2, size=num_samples)  # Binary cancer type labels (0 or 1)

# Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building the neural network
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(num_genes,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Using sigmoid for binary classification
])

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.1)

# Evaluating the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
```

### Explanation of Activation Functions Used

1. **Input Layer**:
   - The input layer receives the gene expression levels of each sample. These are numerical values representing the expression levels of each gene.

2. **Hidden Layers**:
   - **Dense Layer with ReLU Activation**: The first hidden layer has 64 units with ReLU activation. ReLU (`tf.nn.relu`) transforms the input values by setting all negative values to 0, introducing non-linearity to help the model learn complex patterns.
   - **Dense Layer with ReLU Activation**: The second hidden layer has 32 units with ReLU activation, further transforming the data.

3. **Output Layer**:
   - **Dense Layer with Sigmoid Activation**: The output layer has 1 unit with sigmoid activation (`tf.nn.sigmoid`). The sigmoid function maps the output to a range between 0 and 1, which is suitable for binary classification tasks. It outputs the probability of the sample belonging to class 1 (e.g., cancer type B).

### Importance in Gene Expression Data

- **Non-linearity**: Activation functions like ReLU introduce non-linearity, which allows the neural network to model complex relationships between gene expression levels and cancer types.
- **Probability Output**: The sigmoid activation in the output layer is crucial for binary classification tasks, providing a probability score that can be interpreted as the likelihood of a sample belonging to a specific cancer type.

### Summary

Activation functions play a vital role in transforming gene expression data through the layers of a neural network. ReLU is typically used in hidden layers to handle non-linearities, while sigmoid (or softmax for multi-class classification) is used in the output layer to produce probability scores for classification tasks. This example demonstrates how you can build and train a neural network to classify cancer types based on gene expression data, leveraging activation functions to enhance the model's learning capabilities.
