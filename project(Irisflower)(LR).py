import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
# Load data
df = pd.read_csv(r"C:\Users\DOAN SINH DUC\Downloads\archive\IRIS.csv")

# Tạo dataframe
df = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']]
# Chuyển thành mảng trong thư viện Numpy
x = np.asarray(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
y = np.asarray(df['species'])
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Chia dữ liệu thành tập train và tập test
x_train, x_test, y_train, y_test = train_test_split(x,y_encoded, test_size = 0.2, random_state = 42, stratify = y)

# Chuẩn hóa dữ liệu
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Tạo mô hình LR
LR = LogisticRegression(C = 30, solver = 'lbfgs').fit(x_train, y_train)

# Dự đoán giá trị
yhat = LR.predict(x_test)
yhat_proba = LR.predict_proba(x_test)
loss = log_loss(y_test, yhat_proba) 
print("Value of loss function: ", loss)
# Tính Accuracy
accuracy = accuracy_score(y_test, yhat)
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
        new_data = scaler.transform(new_data)
        
        # Dự đoán nhãn
        prediction = LR.predict(new_data)
        y_predicted = label_encoder.inverse_transform(prediction)[0]
        print("Dự đoán nhãn:", y_predicted)
        
        # Dự đoán xác suất
        prediction_prob = LR.predict_proba(new_data)
        print("Xác suất:", prediction_prob)
        
        # Hỏi người dùng có tiếp tục không
        if input("Do you want to predict more? (Y or N)").strip().upper() == 'N':
            print("End program")
            break
else:
    print("End program")
