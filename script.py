import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Preprocess Function (Provided)
def preprocess():
    mat = loadmat('mnist_all.mat')
    n_feature = mat.get("train1").shape[1]
    n_sample = sum([mat.get(f"train{i}").shape[0] for i in range(10)])
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    validation_data = np.zeros((10 * n_validation, n_feature))
    validation_label = np.zeros((10 * n_validation, 1))

    for i in range(10):
        data = mat.get(f"train{i}")
        validation_data[i * n_validation:(i + 1) * n_validation, :] = data[:n_validation, :]
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i

    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0

    for i in range(10):
        data = mat.get(f"train{i}")
        size_i = data.shape[0]
        train_data[temp:temp + size_i - n_validation, :] = data[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i
        temp += size_i - n_validation

    n_test = sum([mat.get(f"test{i}").shape[0] for i in range(10)])
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))

    temp = 0
    for i in range(10):
        data = mat.get(f"test{i}")
        size_i = data.shape[0]
        test_data[temp:temp + size_i, :] = data
        test_label[temp:temp + size_i, :] = i
        temp += size_i

    sigma = np.std(train_data, axis=0)
    valid_features = np.where(sigma > 0.001)[0]

    train_data = train_data[:, valid_features] / 255.0
    validation_data = validation_data[:, valid_features] / 255.0
    test_data = test_data[:, valid_features] / 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label

# Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Binary Logistic Regression Objective Function
def blrObjFunction(initialWeights, *args):
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_features = train_data.shape[1]

    # Add bias term
    train_data_bias = np.hstack((np.ones((n_data, 1)), train_data))
    w = initialWeights.reshape((n_features + 1, 1))
    theta = sigmoid(train_data_bias @ w)

    # Cross-entropy error
    error = -np.sum(labeli * np.log(theta + 1e-10) + (1 - labeli) * np.log(1 - theta + 1e-10)) / n_data
    error_grad = (train_data_bias.T @ (theta - labeli)).flatten() / n_data

    return error, error_grad

# Binary Logistic Regression Prediction
def blrPredict(W, data):
    data_bias = np.hstack((np.ones((data.shape[0], 1)), data))
    predictions = sigmoid(data_bias @ W)
    return np.argmax(predictions, axis=1).reshape(-1, 1)

# Multi-class Logistic Regression Objective Function
def mlrObjFunction(params, *args):
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    train_data_bias = np.hstack((np.ones((n_data, 1)), train_data))
    W = params.reshape((n_features + 1, 10))

    theta = np.exp(train_data_bias @ W)
    theta = theta / np.sum(theta, axis=1, keepdims=True)

    error = -np.sum(labeli * np.log(theta + 1e-10)) / n_data
    error_grad = (train_data_bias.T @ (theta - labeli)).flatten() / n_data

    return error, error_grad

# Multi-class Logistic Regression Prediction
def mlrPredict(W, data):
    data_bias = np.hstack((np.ones((data.shape[0], 1)), data))
    theta = np.exp(data_bias @ W)
    theta = theta / np.sum(theta, axis=1, keepdims=True)
    return np.argmax(theta, axis=1).reshape(-1, 1)

# Main Logic for Training and Testing
if __name__ == "__main__":
    train_data, train_label, val_data, val_label, test_data, test_label = preprocess()

    # Binary Logistic Regression Training
    n_class = 10
    n_feature = train_data.shape[1]
    W = np.zeros((n_feature + 1, n_class))

    for i in range(n_class):
        labeli = (train_label == i).astype(int).flatten()
        args = (train_data, labeli)
        initialWeights = np.zeros(n_feature + 1)
        opts = {'maxiter': 100}
        result = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
        W[:, i] = result.x

    # Predict using Binary Logistic Regression
    predicted_label = blrPredict(W, test_data)
    print(f"Binary Logistic Regression Accuracy: {np.mean(predicted_label == test_label) * 100:.2f}%")

    # SVM Training and Evaluation
    svm_model = SVC(kernel='rbf', gamma=0.05, C=10)
    svm_model.fit(train_data, train_label.ravel())
    svm_accuracy = svm_model.score(test_data, test_label.ravel())
    print(f"SVM Accuracy: {svm_accuracy * 100:.2f}%")
