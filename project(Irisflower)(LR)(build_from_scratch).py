import pandas as pd
import numpy as np

# Hàm chia dữ liệu
def train_split_test(X, y, test_size, random_state):
    amount = X.shape[0]
    test_amount = int(amount * test_size)
    np.random.seed(random_state)
    index_array = np.arange(amount)
    np.random.shuffle(index_array)  # Đảo vị trí các phần tử trong mảng không tạo mảng mới
    X_train, X_test = X[index_array[test_amount:]], X[index_array[:test_amount]]
    y_train, y_test = y[index_array[test_amount:]], y[index_array[:test_amount]]
    return X_train, X_test, y_train, y_test

# Hàm chuẩn hóa dữ liệu
def standardscaler(x, train_mean=None, train_std=None): 
    if train_mean is None or train_std is None:
        return (x - np.mean(x, axis=0)) / x.std(axis=0)
    return (x - train_mean) / train_std

# Hàm Softmax (Dùng cho phân loại đa lớp)
def softmax(x):
    if x.ndim == 1:  # Nếu x là mảng 1 chiều, chuyển thành mảng 2 chiều
        x = x.reshape(1, -1)
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# Hàm Log Loss
def log_loss(y_test, y_prob):
    m = y_prob.shape[0]  # Số mẫu
    K = y_prob.shape[1]  # Số lớp
    y_one_hot = np.zeros((m, K))  # Khởi tạo ma trận one-hot encoding
    y_test = y_test.astype(int)  # Đảm bảo y_test là số nguyên
    y_one_hot[np.arange(m), y_test] = 1  # Gán 1 ở vị trí tương ứng lớp thật
    log_loss_value = -np.sum(y_one_hot * np.log(y_prob + 1e-15)) / m  # Thêm epsilon để tránh log(0)
    return log_loss_value

# Hàm huấn luyện Logistic Regression
def logistic_regression(X, y, learning_rate, epoch, num_classes):
    # DÙng SGD
    m, n = X.shape
    w = np.zeros((n, num_classes))  # Ma trận trọng số cho các lớp
    for j in range(epoch):
        for i in range(m):
            z = np.dot(X[i], w)  # Tính giá trị tuyến tính (logit) cho tất cả các lớp
            y_prob = softmax(z.reshape(1, -1))  # Tính xác suất với Softmax
            y_one_hot = np.zeros(num_classes)  # One-hot encoding cho nhãn thực tế
            y_one_hot[y[i]] = 1  # Gán 1 ở vị trí nhãn thực tế
            dz = y_prob - y_one_hot  # Sai số
            dw = np.outer(X[i], dz)  # Gradient cho từng lớp
            w -= learning_rate * dw  # Cập nhật trọng số
    return w

# Hàm dự đoán
def predict(X, w):
    z = np.dot(X, w)  # Tính giá trị tuyến tính
    y_prob = softmax(z)  # Tính xác suất
    y_predict = np.argmax(y_prob, axis=1)  # Lấy nhãn dự đoán
    return y_predict, y_prob

# Load dữ liệu
df = pd.read_csv(r"C:\Users\DOAN SINH DUC\Downloads\archive\IRIS.csv")

# Tạo dataframe
X = np.asarray(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
y = np.asarray(df['species'])

# Kiểm tra nhãn và mã hóa
encoder = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y_encoded = np.array(list(map(encoder.get, y)))

# Tiền xử lý
X_train, X_test, y_train, y_test = train_split_test(X, y_encoded, test_size=0.2, random_state=42)
train_mean, train_std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
X_train = standardscaler(X_train)
X_test = standardscaler(X_test, train_mean=train_mean, train_std=train_std)

# Huấn luyện mô hình Logistic Regression
num_classes = 3  # Số lớp
w = logistic_regression(X_train, y_train, learning_rate=0.01, epoch=1000, num_classes=num_classes)

# Dự đoán trên tập kiểm tra
y_pred, y_prob = predict(X_test, w)

# Tính Log Loss
loss = log_loss(y_test, y_prob)
print("Log Loss:", loss)

# Độ chính xác
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)

# Đánh giá thử 1 mẫu 
print("Do you want to predict one? (Y or N)")
if input().strip().upper() == 'Y':
    while True:
        # Nhập giá trị từ bàn phím
        sepal_length = float(input("Enter sepal_length: "))
        sepal_width = float(input("Enter sepal_width: "))
        petal_length = float(input("Enter petal_length: "))
        petal_width = float(input("Enter petal_width: "))
        
        # Xây dựng dữ liệu mới thành mảng
        new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Chuẩn hóa dữ liệu
        new_data = standardscaler(new_data, train_mean=train_mean, train_std=train_std)
        
        # Dự đoán nhãn
        prediction, prediction_prob = predict(new_data, w)
        predicted_label = list(encoder.keys())[list(encoder.values()).index(prediction[0])]
        print("Dự đoán nhãn:", predicted_label)
        print("Xác suất:", prediction_prob)

        # Hỏi người dùng có tiếp tục không
        if input("Do you want to predict more? (Y or N)").strip().upper() == 'N':
            print("End program")
            break
else:
    print("End program")
