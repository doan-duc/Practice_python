import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Hàm chuẩn hóa dữ liệu
def train_split_test(X, y, test_size, random_state):
    amount = X.shape[0]
    test_amount = int(amount * test_size)
    np.random.seed(random_state)
    index_array = np.arange(amount) # Mảng các chỉ số của các phần tửtử
    np.random.shuffle(index_array)  # Đảo vị trí các phần tử trong mảng không tạo mảng mới
    X_train, X_test = X[index_array[test_amount:]], X[index_array[:test_amount]]
    y_train, y_test = y[index_array[test_amount:]], y[index_array[:test_amount]]
    return X_train, X_test, y_train, y_test
def standardscaler(x, train_mean=None, train_std=None): 
    if train_mean is None or train_std is None:
        return (x - np.mean(x, axis=0)) / x.std(axis=0)
    return (x - train_mean) / train_std

# Hàm Log Loss
def log_loss(y_true, y_prob):
    m = y_true.size
    eps = 1e-15
    return -(y_true * np.log(y_prob + eps) + (1 - y_true) * np.log(1 - y_prob + eps)).mean()
# Các hàm train Logistic Regression
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def train_logistic_gd(X, y, lr=0.1, epoch=5000):
    w = np.zeros(X.shape[1])          # vector (n_features,)
    for i in range(epoch):
        z = X @ w                      # (m,)
        y_hat = sigmoid(z)             # (m,)
        grad = X.T @ (y_hat - y) / y.size
        w -= lr * grad
    return w
def predict_prob(X, w):
    return sigmoid(X @ w)
def predict_label(X, w):
    probs = predict_prob(X, w)        # mảng shape (m,)
    labels = []                        # tạo list labels
    for p in probs:
        if p >= 0.5:
            labels.append(1)
        else:
            labels.append(0)
    return np.array(labels, dtype=int)

# 1. Đọc dữ liệu
df = pd.read_csv(r"C:\Users\DOAN SINH DUC\Downloads\ChurnData.csv")

# 2. Chọn dữ liệu
x = df.drop(columns='churn').values
y = df['churn'].values
# Tiền xử lý
X_train, X_test, y_train, y_test = train_split_test(x, y, test_size=0.4, random_state=42)
train_mean, train_std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
X_train = standardscaler(X_train)
X_test = standardscaler(X_test, train_mean=train_mean, train_std=train_std)

# Huấn luyện mô hình Logistic Regression
w = train_logistic_gd(X_train, y_train, lr=0.01, epoch=500)

y_prob = predict_prob(X_test, w)

# Dự đoán nhãn
y_pred = predict_label(X_test, w)


# Tính Log Loss
loss = log_loss(y_test, y_prob)
print("Log Loss:", loss)

# Dự đoán nhãn trên tập train
y_pred_train = predict_label(X_train, w)

# Tính accuracy trên tập train
accuracy_train = np.mean(y_pred_train == y_train)
print("Train Accuracy:", accuracy_train)

# Tính accuracy trên tập test (như trước)
accuracy_test = np.mean(y_pred == y_test)
print("Test Accuracy:", accuracy_test)
