Certainly! Here’s a set of exercises designed to help you practice and reinforce your TensorFlow fundamentals:

### 1. **Basic Tensor Operations**

- **Exercise 1.1**: Create a TensorFlow tensor with a shape of `(3, 3)` and initialize it with random values. Perform basic operations such as addition, subtraction, multiplication, and division on this tensor.

    ```python
    import tensorflow as tf
    
    tensor = tf.random.uniform((3, 3))
    
    # Basic operations
    add_result = tensor + 1
    sub_result = tensor - 1
    mul_result = tensor * 2
    div_result = tensor / 2
    ```
  
- **Exercise 1.2**: Create a tensor with a shape of `(4, 4)` and use TensorFlow functions to reshape it into a `(2, 8)` tensor. Verify the reshaped tensor’s shape.

    ```python
    import tensorflow as tf
    
    tensor = tf.random.uniform((4, 4))
    
    # Reshape
    reshaped_tensor = tf.reshape(tensor, (2, 8))
    ```

- **Exercise 1.3**: Implement broadcasting by adding a 1D tensor of shape `(3,)` to a 2D tensor of shape `(3, 3)`.
  
    ```python
    import tensorflow as tf
    
    tensor_2d = tf.random.uniform((3, 3))
    tensor_1d = tf.constant([1, 2, 3])
  
    # Broadcasting
    result = tensor_2d + tensor_1d
    ```
    
- **Exercise 1.4**: Create a 3D tensor of shape `(2, 3, 4)` with random values and compute its mean along the first axis.
  
    ```python
    import tensorflow as tf
    
    tensor = tf.random.uniform((2, 3, 4))
    mean_result = tf.reduce_mean(tensor, axis=0)
    print(mean_result)
    ```
  
- **Exercise 1.5**: Create two tensors of shape `(5, 5)` and perform element-wise subtraction. Ensure that both tensors are of the same data type.
  
    ```python
    import tensorflow as tf
    
    tensor1 = tf.random.uniform((5, 5))
    tensor2 = tf.random.uniform((5, 5))
    sub_result = tensor1 - tensor2
    print(sub_result)
    ```
    
- **Exercise 1.6**: Create a tensor of shape `(3, 3)` initialized with zeros and another tensor of the same shape initialized with ones. Perform element-wise multiplication.

    ```python
    import tensorflow as tf
    
    zeros_tensor = tf.zeros((3, 3))
    ones_tensor = tf.ones((3, 3))
    mul_result = zeros_tensor * ones_tensor
    print(mul_result)
    ```

- **Exercise 1.7**: Create a tensor of shape `(4, 4)` with random integers between 1 and 10. Find the maximum value in each row.
    ```python
    import tensorflow as tf
    
    tensor = tf.random.uniform((4, 4), minval=1, maxval=10, dtype=tf.int32)
    max_per_row = tf.reduce_max(tensor, axis=1)
    print(max_per_row)
    ```
    
- **Exercise 1.8**: Create a tensor of shape `(3, 3)` and transpose it. Verify the shape of the transposed tensor.

    ```python
    import tensorflow as tf
    
    tensor = tf.random.uniform((3, 3))
    transposed_tensor = tf.transpose(tensor)
    print(transposed_tensor)
    ```

- **Exercise 1.9**: Create a tensor of shape `(2, 2, 2)` and compute the sum of elements across the last axis.

    ```python
    import tensorflow as tf
    
    tensor = tf.random.uniform((2, 2, 2))
    sum_result = tf.reduce_sum(tensor, axis=-1)
    print(sum_result)
    ```

- **Exercise 1.10**: Create a tensor with shape `(4, 4)` and use slicing to extract a sub-tensor of shape `(2, 2)` from the top-left corner.

    ```python
    import tensorflow as tf
    
    tensor = tf.random.uniform((4, 4))
    sub_tensor = tensor[:2, :2]
    print(sub_tensor)
    ```

- **Exercise 1.11**: Create a tensor of shape `(2, 3)` and use `tf.concat` to concatenate it with another tensor of shape `(2, 2)` along the second axis. Ensure the resulting tensor's shape is correct.

    ```python
    import tensorflow as tf
    
    tensor1 = tf.random.uniform((2, 3))
    tensor2 = tf.random.uniform((2, 2))
    concat_result = tf.concat([tensor1, tensor2], axis=1)
    print(concat_result)
    ```
    
- **Exercise 1.12**: Create a tensor of shape `(5, 5)` and perform a matrix multiplication with itself. Verify the shape of the resulting tensor.

    ```python
    import tensorflow as tf
    
    tensor = tf.random.uniform((5, 5))
    matrix_mul_result = tf.matmul(tensor, tensor)
    print(matrix_mul_result)
    ```
    
- **Exercise 1.13**: Create a tensor of shape `(4, 4)` and normalize its values to the range `[0, 1]` using min-max normalization.

    ```python
    import tensorflow as tf
    
    tensor = tf.random.uniform((4, 4))
    min_val = tf.reduce_min(tensor)
    max_val = tf.reduce_max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    print(normalized_tensor)
    ```
  
- **Exercise 1.14**: Create a TensorFlow tensor with a shape of `(4, 4)` and initialize it with random values between 0 and 1. Slice the tensor to get the bottom-right 2x2 sub-tensor.

    ```python
    import tensorflow as tf

    tensor = tf.random.uniform((4, 4), minval=0, maxval=1)
    sliced_tensor = tensor[2:4, 2:4]
    print(sliced_tensor)
    ```

- **Exercise 1.15**: Create a 1-D tensor with 10 elements using `tf.range`. Reshape this tensor into a 2x5 tensor.

    ```python
    tensor = tf.range(10)
    reshaped_tensor = tf.reshape(tensor, (2, 5))
    print(reshaped_tensor)
    ```

- **Exercise 1.16**: Create a tensor of shape `(5, 5)` filled with ones. Change the center element to 0.

    ```python
    tensor = tf.ones((5, 5))
    tensor = tf.tensor_scatter_nd_update(tensor, indices=[[2, 2]], updates=[0])
    print(tensor)
    ```

- **Exercise 1.17**: Create two 1-D tensors of length 5 with random values. Concatenate them to create a 1-D tensor of length 10.
    
    ```python
    tensor1 = tf.random.uniform((5,))
    tensor2 = tf.random.uniform((5,))
    concatenated_tensor = tf.concat([tensor1, tensor2], axis=0)
    print(concatenated_tensor)
    ```

- **Exercise 1.18**: Create a 3x3 identity matrix using TensorFlow.

    ```python
    identity_matrix = tf.eye(3)
    print(identity_matrix)
    ```

- **Exercise 1.19**: Create a tensor of shape `(2, 3)` with random values. Calculate the mean and standard deviation of the tensor.

    ```python
    tensor = tf.random.uniform((2, 3))
    mean = tf.reduce_mean(tensor)
    stddev = tf.math.reduce_std(tensor)
    print("Mean:", mean.numpy())
    print("Standard Deviation:", stddev.numpy())
    ```

- **Exercise 1.20**: Create a tensor of shape `(3, 3)` with random values. Normalize the tensor (subtract the mean and divide by the standard deviation).
 
    ```python
    tensor = tf.random.uniform((3, 3))
    mean, stddev = tf.reduce_mean(tensor), tf.math.reduce_std(tensor)
    normalized_tensor = (tensor - mean) / stddev
    print(normalized_tensor)
    ```

- **Exercise 1.21**: Create a tensor of shape `(4, 4)` with random values between -1 and 1. Apply the ReLU activation function to this tensor.

    ```python
    tensor = tf.random.uniform((4, 4), minval=-1, maxval=1)
    relu_tensor = tf.nn.relu(tensor)
    print(relu_tensor)
    ```

- **Exercise 1.22**: Create a tensor of shape `(5, 5)` with normally distributed random values. Calculate the sum of all elements in the tensor.

    ```python
    tensor = tf.random.normal((5, 5))
    sum_tensor = tf.reduce_sum(tensor)
    print(sum_tensor)
    ```

- **Exercise 1.23**: Create two tensors of shape `(2, 2)` with random values. Perform element-wise multiplication and matrix multiplication on these tensors.

    ```python
    tensor1 = tf.random.uniform((2, 2))
    tensor2 = tf.random.uniform((2, 2))
    elementwise_mul = tf.multiply(tensor1, tensor2)
    matrix_mul = tf.matmul(tensor1, tensor2)
    print("Element-wise multiplication:\n", elementwise_mul)
    print("Matrix multiplication:\n", matrix_mul)
    ```

- **Exercise 1.24**: Create a tensor of shape `(3, 4)` with random values. Transpose this tensor.

    ```python
    tensor = tf.random.uniform((3, 4))
    transposed_tensor = tf.transpose(tensor)
    print(transposed_tensor)
    ```

- **Exercise 1.25**: Create a tensor of shape `(2, 3, 4)` with random values. Reshape this tensor into a 2-D tensor of shape `(6, 4)`.

    ```python
    tensor = tf.random.uniform((2, 3, 4))
    reshaped_tensor = tf.reshape(tensor, (6, 4))
    print(reshaped_tensor)
    ```

- **Exercise 1.26**: Create a tensor of shape `(3, 3)` with random values. Extract the diagonal elements of this tensor.

    ```python
    tensor = tf.random.uniform((3, 3))
    diagonal = tf.linalg.diag_part(tensor)
    print(diagonal)
    ```

- **Exercise 1.27**: Create a tensor of shape `(4, 4)` with values from 1 to 16. Reverse the order of elements in each row.

    ```python
    tensor = tf.reshape(tf.range(1, 17), (4, 4))
    reversed_tensor = tf.reverse(tensor, axis=[1])
    print(reversed_tensor)
    ```

- **Exercise 1.28**: Create a tensor of shape `(4, 4)` with values from 1 to 16. Reverse the order of elements in each column.

    ```python
    tensor = tf.reshape(tf.range(1, 17), (4, 4))
    reversed_tensor = tf.reverse(tensor, axis=[0])
    print(reversed_tensor)
    ```

- **Exercise 1.29**: Create a tensor of shape `(5, 5)` with random values. Calculate the determinant of this tensor.

    ```python
    tensor = tf.random.uniform((5, 5))
    determinant = tf.linalg.det(tensor)
    print(determinant)
    ```

- **Exercise 1.30**: Create a tensor of shape `(3, 3)` with random values. Calculate the inverse of this tensor (if it's invertible).

    ```python
    tensor = tf.random.uniform((3, 3))
    inverse_tensor = tf.linalg.inv(tensor)
    print(inverse_tensor)
    ```

- **Exercise 1.31**: Create a tensor of shape `(4, 4)` with values from 1 to 16. Flatten this tensor into a 1-D tensor.

    ```python
    tensor = tf.reshape(tf.range(1, 17), (4, 4))
    flattened_tensor = tf.reshape(tensor, [-1])
    print(flattened_tensor)
    ```

- **Exercise 1.32**: Create a tensor of shape `(3, 3, 3)` with random values. Slice this tensor to get the first two elements along the first dimension.

    ```python
    tensor = tf.random.uniform((3, 3, 3))
    sliced_tensor = tensor[:2, :, :]
    print(sliced_tensor)
    ```

- **Exercise 1.33**: Create a tensor of shape `(6, 6)` with random values. Select and print every other row and column.

    ```python
    tensor = tf.random.uniform((6, 6))
    sliced_tensor = tensor[::2, ::2]
    print(sliced_tensor)
    ```

- **Exercise 1.34**: Create a tensor of shape `(3, 3, 3)` with random values. Sum the tensor along the second dimension.

    ```python
    tensor = tf.random.uniform((3, 3, 3))
    summed_tensor = tf.reduce_sum(tensor, axis=1)
    print(summed_tensor)
    ```

- **Exercise 1.35**: Create a tensor of shape `(4, 4)` with random values. Split this tensor into four 2x2 sub-tensors.

    ```python
    tensor = tf.random.uniform((4, 4))
    sub_tensors = tf.split(tensor, num_or_size_splits=2, axis=0)
    sub_tensors = [tf.split(sub_tensor, num_or_size_splits=2, axis=1) for sub_tensor in sub_tensors]
    print(sub_tensors[0][0])
    print(sub_tensors[0][1])
    print(sub_tensors[1][0])
    print(sub_tensors[1][1])
    ```

- **Exercise 1.36**: Create a tensor of shape `(3, 4)` with random values. Stack this tensor with itself along a new dimension to create a tensor of shape `(2, 3, 4)`.

    ```python
    tensor = tf.random.uniform((3, 4))
    stacked_tensor = tf.stack([tensor, tensor], axis=0)
    print(stacked_tensor)
    ```

- **Exercise 1.37**: Create two tensors of shape `(3, 3)` with random values. Compute the element-wise maximum and minimum of these tensors.

    ```python
    tensor1 = tf.random.uniform((3, 3))
    tensor2 = tf.random.uniform((3, 3))
    max_tensor = tf.maximum(tensor1, tensor2)
    min_tensor = tf.minimum(tensor1, tensor2)
    print("Element-wise maximum:\n", max_tensor)
    print("Element-wise minimum:\n", min_tensor)
    ```

- **Exercise 1.38**: Create a tensor of shape `(4, 4)` with random values. Clip the values in this tensor to be between 0.2 and 0.8.

    ```python
    tensor = tf.random.uniform((4, 4))
    clipped_tensor = tf.clip_by_value(tensor, clip_value_min=0.2, clip_value_max=0.8)
    print(clipped_tensor)
    ```

- **Exercise 1.39**: Create two tensors of shape `(3, 4)` with random values. Perform matrix multiplication of these tensors with another tensor of shape `(4, 5)` using broadcasting.

    ```python
    import tensorflow as tf
    
    # Tensors
    a = tf.random.uniform((2, 3, 4))
    b = tf.random.uniform((4, 5))
    
    # Matrix multiplication with broadcasting
    result = tf.matmul(a, b)
    
    print(result)
    ```
  
- **Exercise 1.40**: Create a tensor of shape `(4, 4, 4)` with random values. Transpose this tensor, swapping the second and third dimensions.

    ```python
    import tensorflow as tf
    
    # Tensor
    tensor = tf.random.uniform((2, 3, 4, 5))
    
    # Transpose
    transposed = tf.transpose(tensor, perm=[0, 2, 1, 3])
    
    print(transposed)
    ```

- **Exercise 1.41**: Create two tensors of shape `(3, 4)` with random values. Compute the element-wise maximum of these tensors.

    ```python
    import tensorflow as tf
    
    # Tensors
    a = tf.random.uniform((3, 4))
    b = tf.random.uniform((3, 4))
    
    # Element-wise maximum
    maximum = tf.maximum(a, b)
    
    print(maximum)
    ```

- **Exercise 1.42**: Create a tensor of shape `(5, 5)` with random values. Slice this tensor to get a `(2, 2)` sub-tensor from the center and assign new values to this sliced part.

    ```python
    import tensorflow as tf
    
    # Tensor
    tensor = tf.Variable(tf.random.uniform((5, 5)))
    
    # Slice and assign
    tensor[1:3, 2:4].assign(tf.ones((2, 2)))
    
    print(tensor)
    ```
    
- **Exercise 1.43**: Define a function \( z = x^2 + y^2 \). Create two variables `x` and `y` initialized with random values. Compute the gradient of \( z \) with respect to `x` and `y`.

    ```python
    import tensorflow as tf
    
    # Variables
    x = tf.Variable(3.0)
    y = tf.Variable(4.0)
    
    # Function
    with tf.GradientTape() as tape:
        z = x**2 + y**2
    
    # Compute gradient
    gradients = tape.gradient(z, [x, y])
    
    print(gradients)
    ```
    
- **Exercise 1.44**: Create a tensor of shape `(4, 4, 4)` with random values. Reshape this tensor to have a shape `(8, -1)`, inferring the second dimension.

    ```python
    import tensorflow as tf
    
    # Tensor
    tensor = tf.random.uniform((4, 4, 4))
    
    # Reshape with inferred dimension
    reshaped = tf.reshape(tensor, (-1, 8))
    
    print(reshaped)
    ```

- **Exercise 1.45**: Create a tensor of integer indices `[0, 1, 2, 1]`. Convert these indices to a one-hot encoded tensor with a depth of 4.

    ```python
    import tensorflow as tf
    
    # Integer indices
    indices = tf.constant([0, 1, 2, 1])
    
    # One-hot encoding
    one_hot = tf.one_hot(indices, depth=3)
    
    print(one_hot)
    ```

- **Exercise 1.46**: Create a batch of 3 matrices, each of shape `(4, 4)`, with random values. Compute the inverse of each matrix in the batch.

    ```python
    import tensorflow as tf
    
    # Batch of matrices
    batch_matrices = tf.random.uniform((3, 4, 4))
    
    # Batch matrix inversion
    inverted_matrices = tf.linalg.inv(batch_matrices)
    
    print(inverted_matrices)
    ```

- **Exercise 1.47**: Create two tensors of shape `(3, 4)` with random values. Concatenate these tensors along a new axis.

    ```python
    import tensorflow as tf
    
    # Tensors
    a = tf.random.uniform((3, 4))
    b = tf.random.uniform((3, 4))
    
    # Concatenate along a new axis
    concatenated = tf.stack([a, b], axis=0)
    
    print(concatenated)
    ```

- **Exercise 1.48**: Create a tensor of shape `(4, 4)` with random values. Compute the determinant of this matrix.

    ```python
    import tensorflow as tf
    
    # Matrix
    matrix = tf.random.uniform((4, 4))
    
    # Compute determinant
    determinant = tf.linalg.det(matrix)
    
    print(determinant)
    ```

- **Exercise 1.49**: Create a batch of 3 positive-definite matrices, each of shape `(4, 4)`, with random values. Perform Cholesky decomposition on each matrix in the batch.

    ```python
    import tensorflow as tf
    
    # Batch of positive-definite matrices
    batch_matrices = tf.random.uniform((3, 4, 4))
    batch_matrices = tf.matmul(batch_matrices, batch_matrices, transpose_b=True)
    
    # Cholesky decomposition
    cholesky_decomp = tf.linalg.cholesky(batch_matrices)
    
    print(cholesky_decomp)
    ```

- **Exercise 1.50**: Create a tensor of shape `(3, 4)` with random values. Compute the cumulative sum of this tensor along the second axis.

    ```python
    import tensorflow as tf
    
    # Tensor
    tensor = tf.random.uniform((3, 4))
    
    # Cumulative sum along axis 1
    cumsum = tf.math.cumsum(tensor, axis=1)
    
    print(cumsum)
    ```
    
### 2. **Building a Simple Neural Network**

- **Exercise 2.1**: Create a simple neural network model using TensorFlow’s Keras API with the following architecture:
  - Input layer: Shape of 784 (e.g., for MNIST digits)
  - Two dense layers: Each with 128 units and ReLU activation
  - Output layer: 10 units with softmax activation
  - Compile the model with appropriate loss, optimizer, and metrics.

- **Exercise 2.2**: Train the model from Exercise 2.1 on the MNIST dataset for 5 epochs and evaluate its performance on the test set.

- **Exercise 2.3**: Modify the model to include dropout layers after each dense layer to prevent overfitting. Compare the performance with and without dropout.

### 3. **Data Preprocessing**

- **Exercise 3.1**: Load the CIFAR-10 dataset and preprocess it by normalizing pixel values to the range `[0, 1]`. Split the data into training and testing sets.

- **Exercise 3.2**: Use TensorFlow’s data pipeline API to create a dataset from the CIFAR-10 images and labels, applying data augmentation techniques such as random flips and rotations.

### 4. **Custom Training Loop**

- **Exercise 4.1**: Implement a custom training loop using TensorFlow’s `tf.GradientTape` to train a neural network model. Compare the results with using the Keras `model.fit` method.

- **Exercise 4.2**: Create custom loss functions and optimizers in TensorFlow. Implement these custom components in your training loop and evaluate the impact on model performance.

### 5. **Model Evaluation and Tuning**

- **Exercise 5.1**: Evaluate a trained model’s performance using different metrics such as precision, recall, and F1-score. Implement confusion matrix visualizations to analyze model predictions.

- **Exercise 5.2**: Perform hyperparameter tuning for a neural network model by experimenting with different learning rates, batch sizes, and network architectures. Use TensorFlow’s `tf.keras.callbacks` to monitor training progress.

### 6. **Advanced Topics**

- **Exercise 6.1**: Implement a convolutional neural network (CNN) for image classification using TensorFlow. Experiment with different architectures, including varying the number of convolutional layers and filters.

- **Exercise 6.2**: Create a generative adversarial network (GAN) using TensorFlow to generate synthetic images. Implement both the generator and discriminator networks and train them adversarially.

- **Exercise 6.3**: Explore transfer learning by using a pre-trained model (e.g., InceptionV3) and fine-tune it for a custom dataset. Evaluate how transfer learning impacts model performance.

### 7. **TensorFlow Serving**

- **Exercise 7.1**: Save a trained TensorFlow model in the SavedModel format and serve it using TensorFlow Serving. Test the model's predictions using REST API calls.

- **Exercise 7.2**: Implement a client application that interacts with TensorFlow Serving to send image data and receive predictions.

Feel free to adapt these exercises based on your current knowledge level and learning goals. Let me know if you need any specific details or assistance with any of the exercises!




Certainly! Here’s a structured answer set for the exercises to help you verify your work and understand the solutions better. If you need help with any specific exercise, feel free to ask!



### 2. **Building a Simple Neural Network**

**Exercise 2.1**:
- **Solution**:
  ```python
  import tensorflow as tf
  from tensorflow.keras import layers, models

  # Define model
  model = models.Sequential([
      layers.Flatten(input_shape=(784,)),
      layers.Dense(128, activation='relu'),
      layers.Dense(128, activation='relu'),
      layers.Dense(10, activation='softmax')
  ])

  # Compile model
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  ```

**Exercise 2.2**:
- **Solution**:
  ```python
  import tensorflow as tf
  from tensorflow.keras.datasets import mnist

  # Load and preprocess data
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0

  # Train model
  model.fit(x_train, y_train, epochs=5)
  
  # Evaluate model
  test_loss, test_acc = model.evaluate(x_test, y_test)
  ```

**Exercise 2.3**:
- **Solution**:
  ```python
  import tensorflow as tf
  from tensorflow.keras import layers, models

  # Define model with dropout
  model = models.Sequential([
      layers.Flatten(input_shape=(784,)),
      layers.Dense(128, activation='relu'),
      layers.Dropout(0.5),
      layers.Dense(128, activation='relu'),
      layers.Dropout(0.5),
      layers.Dense(10, activation='softmax')
  ])

  # Compile model
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  ```

### 3. **Data Preprocessing**

**Exercise 3.1**:
- **Solution**:
  ```python
  import tensorflow as tf
  from tensorflow.keras.datasets import cifar10

  # Load and preprocess data
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  ```

**Exercise 3.2**:
- **Solution**:
  ```python
  import tensorflow as tf
  from tensorflow.keras.preprocessing.image import ImageDataGenerator

  # Create data augmentation pipeline
  datagen = ImageDataGenerator(
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True
  )

  # Apply augmentation to training data
  train_generator = datagen.flow(x_train, y_train, batch_size=32)
  ```

### 4. **Custom Training Loop**

**Exercise 4.1**:
- **Solution**:
  ```python
  import tensorflow as tf

  # Define model
  model = tf.keras.Sequential([
      tf.keras.layers.Flatten(input_shape=(784,)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
  ])

  # Define optimizer and loss
  optimizer = tf.keras.optimizers.Adam()
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

  # Custom training loop
  for epoch in range(5):
      with tf.GradientTape() as tape:
          logits = model(x_train, training=True)
          loss = loss_fn(y_train, logits)
      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
  ```

**Exercise 4.2**:
- **Solution**:
  ```python
  import tensorflow as tf

  # Define custom loss
  def custom_loss(y_true, y_pred):
      return tf.reduce_mean(tf.square(y_true - y_pred))

  # Define custom optimizer
  class CustomOptimizer(tf.keras.optimizers.Optimizer):
      def __init__(self, learning_rate=0.01, name="CustomOptimizer", **kwargs):
          super().__init__(name, **kwargs)
          self.learning_rate = learning_rate

      def _create_slots(self, var_list):
          pass

      def _resource_apply_dense(self, grad, var, apply_state=None):
          return var.assign_sub(grad * self.learning_rate)

  # Apply custom loss and optimizer in training loop
  optimizer = CustomOptimizer(learning_rate=0.01)
  ```

### 5. **Model Evaluation and Tuning**

**Exercise 5.1**:
- **Solution**:
  ```python
  import matplotlib.pyplot as plt
  from sklearn.metrics import classification_report, confusion_matrix

  # Evaluate model
  y_pred = model.predict(x_test)
  y_pred_classes = tf.argmax(y_pred, axis=1)
  y_true = y_test.flatten()

  # Print classification report
  print(classification_report(y_true, y_pred_classes))

  # Plot confusion matrix
  cm = confusion_matrix(y_true, y_pred_classes)
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.colorbar()
  plt.show()
  ```

**Exercise 5.2**:
- **Solution**:
  ```python
  from tensorflow.keras.tuner import RandomSearch

  # Define model builder function
  def build_model(hp):
      model = tf.keras.Sequential([
          tf.keras.layers.Flatten(input_shape=(784,)),
          tf.keras.layers.Dense(units=hp.Int('units', min_value=64, max_value=256, step=64), activation='relu'),
          tf.keras.layers.Dense(10, activation='softmax')
      ])
      model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-1, sampling='LOG')),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
      return model

  # Instantiate tuner
  tuner = RandomSearch(build_model, objective='val_accuracy', max_trials=10, executions_per_trial=1)

  # Search for best hyperparameters
  tuner.search(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
  ```

### 6. **Advanced Topics**

**Exercise 6.1**:
- **Solution**:
  ```python
  from tensorflow.keras.datasets import cifar10
  from tensorflow.keras import layers, models

  # Define CNN model
  cnn_model = models.Sequential([
      layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(64, (3, 3), activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(64, (3, 3), activation='relu'),
      layers.Flatten(),
      layers.Dense(64, activation='relu'),
      layers.Dense(10, activation='softmax')
  ])

  # Compile and train model
  cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  cnn_model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
  ```

**Exercise 6.2**:
- **Solution**:
  ```python
  from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization
  from tensorflow.keras.models import Sequential

  # Define GAN models
  def build_generator():
      model = Sequential([
          Dense(128, input_dim=100),
          LeakyReLU(0.2),
          Dense(784, activation='sigmoid')
      ])
      return model

  def build_discriminator():
      model = Sequential([
          Dense(128, input_dim=784),
          LeakyReLU(0.2),
          Dense(1, activation='sigmoid')
      ])
      return model
  ```

**Exercise 6.3**:
- **Solution**:
  ```python
  from tensorflow.keras.applications import InceptionV3
  from tensorflow.keras.preprocessing.image import ImageDataGenerator

  # Load pre-trained model
  base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(150,

 150, 3))

  # Freeze base model layers
  base_model.trainable = False

  # Define custom model
  model = models.Sequential([
      base_model,
      layers.GlobalAveragePooling2D(),
      layers.Dense(1, activation='sigmoid')
  ])

  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  ```

### 7. **TensorFlow Serving**

**Exercise 7.1**:
- **Solution**:
  ```bash
  # Save model
  model.save('saved_model/my_model')

  # Run TensorFlow Serving (from command line)
  tensorflow_model_server --rest_api_port=8501 --model_name=my_model --model_base_path="/path/to/saved_model"
  ```

**Exercise 7.2**:
- **Solution**:
  ```python
  import requests
  import json

  # Define the API endpoint
  url = 'http://localhost:8501/v1/models/my_model:predict'

  # Prepare input data
  data = json.dumps({"signature_name": "serving_default", "instances": x_test[:1].tolist()})

  # Make prediction request
  response = requests.post(url, data=data, headers={"Content-Type": "application/json"})
  predictions = response.json()
  ```




