#!/usr/bin/env python
# coding: utf-8

# In[3]:


#数据导入与处理
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(r):
    train_images_path = os.path.join(r, 'train-images-idx3-ubyte')
    train_labels_path = os.path.join(r, 'train-labels-idx1-ubyte')
    test_images_path = os.path.join(r, 't10k-images-idx3-ubyte')
    test_labels_path = os.path.join(r, 't10k-labels-idx1-ubyte')

    X_train, y_train = load_images(train_images_path), load_labels(train_labels_path)
    X_test, y_test = load_images(test_images_path), load_labels(test_labels_path)

    return X_train, y_train, X_test, y_test

def load_images(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28*28) / 255.0

def load_labels(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

r = "C:\\Users\\ad\\Desktop\\data"
X_train, y_train, X_test, y_test = load_data(r)

num_classes = 10
y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]

val_split = 0.1
val_size = int(X_train.shape[0] * val_split)
X_val = X_train[-val_size:]
y_val = y_train[-val_size:]
X_train = X_train[:-val_size]
y_train = y_train[:-val_size]


# In[4]:


#模型构建
class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, hidden_activation='relu'):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.hidden_activation = hidden_activation

        self.initialize_weights()

    def initialize_weights(self):
        self.weights1 = np.random.randn(self.input_size, self.hidden_size1) * np.sqrt(2 / self.input_size)
        self.biases1 = np.zeros((1, self.hidden_size1))

        self.weights2 = np.random.randn(self.hidden_size1, self.hidden_size2) * np.sqrt(2 / self.hidden_size1)
        self.biases2 = np.zeros((1, self.hidden_size2))

        self.weights3 = np.random.randn(self.hidden_size2, self.output_size) * np.sqrt(2 / self.hidden_size2)
        self.biases3 = np.zeros((1, self.output_size))

    def forward(self, X):
        self.layer1_output = np.dot(X, self.weights1) + self.biases1
        self.hidden_output1 = self.relu(self.layer1_output)

        self.layer2_output = np.dot(self.hidden_output1, self.weights2) + self.biases2
        self.hidden_output2 = self.relu(self.layer2_output)

        self.layer3_output = np.dot(self.hidden_output2, self.weights3) + self.biases3
        self.output = self.softmax(self.layer3_output)

        return self.output

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def compute_loss(self, y_true, y_pred, l2_lambda):
        cross_entropy_loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        l2_regularization = 0.5 * l2_lambda * (np.sum(self.weights1 ** 2) + np.sum(self.weights2 ** 2) + np.sum(self.weights3 ** 2))
        return cross_entropy_loss + l2_regularization

    def backward(self, X, y_true, learning_rate=0.01, l2_lambda=0.01):
        output_error = self.output - y_true

        dW3, dB3 = self.compute_gradient(output_error, self.hidden_output2, l2_lambda, self.weights3)
        dW2, dB2 = self.compute_gradient(np.dot(output_error, self.weights3.T) * self.relu_derivative(self.layer2_output), self.hidden_output1, l2_lambda, self.weights2)
        dW1, dB1 = self.compute_gradient(np.dot(np.dot(output_error, self.weights3.T) * self.relu_derivative(self.layer2_output), self.weights2.T) * self.relu_derivative(self.layer1_output), X, l2_lambda, self.weights1)

        self.weights1 -= learning_rate * dW1
        self.biases1 -= learning_rate * dB1
        self.weights2 -= learning_rate * dW2
        self.biases2 -= learning_rate * dB2
        self.weights3 -= learning_rate * dW3
        self.biases3 -= learning_rate * dB3

    def compute_gradient(self, error, input_data, l2_lambda, weights):
        dW = (np.dot(input_data.T, error) / input_data.shape[0]) + (l2_lambda * weights)
        dB = np.sum(error, axis=0, keepdims=True) / input_data.shape[0]
        return dW, dB

def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32, learning_rate=0.01, l2_lambda=0.01):
    n_train = X_train.shape[0]

    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float('inf')
    best_weights_biases = []

    for epoch in range(epochs):
        indices = np.random.permutation(n_train)
        X_train, y_train = X_train[indices], y_train[indices]

        for i in range(0, n_train, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            y_pred = model.forward(X_batch)

            loss = model.compute_loss(y_batch, y_pred, l2_lambda)

            model.backward(X_batch, y_batch, learning_rate=learning_rate, l2_lambda=l2_lambda)

        y_val_pred = model.forward(X_val)
        val_loss = model.compute_loss(y_val, y_val_pred, l2_lambda)
        val_accuracy = np.sum(np.argmax(y_val, axis=1) == np.argmax(y_val_pred, axis=1)) / y_val.shape[0]

        train_losses.append(loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights_biases = [(model.weights1.copy(), model.biases1.copy()),
                                   (model.weights2.copy(), model.biases2.copy()),
                                   (model.weights3.copy(), model.biases3.copy())]

    model.weights1, model.biases1 = best_weights_biases[0]
    model.weights2, model.biases2 = best_weights_biases[1]
    model.weights3, model.biases3 = best_weights_biases[2]


# In[14]:


#训练模型
import matplotlib.pyplot as plt

def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32, learning_rate=0.01, l2_lambda=0.01):
    n_train = X_train.shape[0]

    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float('inf')
    best_weights_biases = []

    for epoch in range(epochs):
        indices = np.random.permutation(n_train)
        X_train_shuffled, y_train_shuffled = X_train[indices], y_train[indices]

        for i in range(0, n_train, batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]

            y_pred = model.forward(X_batch)

            loss = model.compute_loss(y_batch, y_pred, l2_lambda)

            model.backward(X_batch, y_batch, learning_rate=learning_rate, l2_lambda=l2_lambda)

        y_val_pred = model.forward(X_val)
        val_loss = model.compute_loss(y_val, y_val_pred, l2_lambda)
        val_accuracy = np.sum(np.argmax(y_val, axis=1) == np.argmax(y_val_pred, axis=1)) / y_val.shape[0]

        train_losses.append(loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights_biases = [(model.weights1.copy(), model.biases1.copy()),
                                   (model.weights2.copy(), model.biases2.copy()),
                                   (model.weights3.copy(), model.biases3.copy())]

    model.weights1, model.biases1 = best_weights_biases[0]
    model.weights2, model.biases2 = best_weights_biases[1]
    model.weights3, model.biases3 = best_weights_biases[2]

    plot_training_results(train_losses, val_losses, val_accuracies)

    return model

def plot_training_results(train_losses, val_losses, val_accuracies):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='loss')
    plt.plot(val_losses, label='validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


# In[15]:


#测试并选择最佳参数
hidden_sizes = [64, 128, 256]
learning_rates = [0.01, 0.1]
l2_lambdas = [0.01, 0.1]
best_val_accuracy = 0
best_model = None

best_hyperparameters = {}

for hidden_size1 in hidden_sizes:
    for hidden_size2 in hidden_sizes:
        for learning_rate in learning_rates:
            for l2_lambda in l2_lambdas:
                print(f"Training with hidden_size1={hidden_size1}, hidden_size2={hidden_size2}, learning_rate={learning_rate}, l2_lambda={l2_lambda}")

                model = NeuralNetwork(input_size=784, hidden_size1=hidden_size1, hidden_size2=hidden_size2, output_size=10, hidden_activation='relu')
                model = train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, learning_rate=learning_rate, l2_lambda=l2_lambda)

                y_val_pred = model.forward(X_val)
                val_accuracy = np.sum(np.argmax(y_val, axis=1) == np.argmax(y_val_pred, axis=1)) / y_val.shape[0]
                print(f"Validation Accuracy: {val_accuracy:.4f}")

                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_model = model
                    best_hyperparameters = {
                        "hidden_size1": hidden_size1,
                        "hidden_size2": hidden_size2,
                        "learning_rate": learning_rate,
                        "l2_lambda": l2_lambda
                    }

print("Best Hyperparameters:")
print(best_hyperparameters)


# In[ ]:





# In[18]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

# 计算混淆矩阵
conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_test_pred, axis=1))

# 绘制混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# In[19]:


# 找出错误分类的样本
incorrect_indices = np.where(np.argmax(y_test, axis=1) != np.argmax(y_test_pred, axis=1))[0]
incorrect_images = X_test[incorrect_indices]
true_labels = np.argmax(y_test[incorrect_indices], axis=1)
predicted_labels = np.argmax(y_test_pred[incorrect_indices], axis=1)

# 绘制错误分类的样本
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(incorrect_images[i].reshape(28, 28), cmap='gray')
    plt.title(f'True: {true_labels[i]}, Predicted: {predicted_labels[i]}')
    plt.axis('off')
plt.tight_layout()
plt.show()


# In[ ]:




