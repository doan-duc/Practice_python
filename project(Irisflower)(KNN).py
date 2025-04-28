# Import các thư viện cần thiết
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Tải bộ dữ liệu
df = pd.read_csv(r"C:\Users\DOAN SINH DUC\Downloads\archive\IRIS.csv")

# Tạo dataframe và chuẩn hóa
X = np.asarray(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
y = np.asarray(df['species'])
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Chia mô hình
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y)

# Chuẩn hóa dữ liệu đầu vào
scaler = StandardScaler().fit(X_train)  
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Tạo mô hình
neigh = KNeighborsClassifier(n_neighbors = 3).fit(X_train, y_train)

# Dự đoán                                                                                                                                                                                                                                                                                                                           
yhat = neigh.predict(X_test) 
print("Train set Accuracy: ", accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", accuracy_score(y_test, yhat))
# Dự đoán thử 1 mẫu nhập từ bàn phím
print("Do you want to predict one? (Y or N)")
if input().strip().upper() == 'Y':
    while True:
        # Nhập giá trị từ bàn phím
        sepal_length = float(input("Enter sepal_length: "))
        sepal_width = float(input("Enter sepal_width: "))
        petal_length = float(input("Enter petal_length: "))
        petal_width = float(input("Enter petal_width: "))
        
        # Xây dựng dữ liệu mới thành mảng, chuẩn hóa
        new_data = np.asarray([[sepal_length, sepal_width, petal_length, petal_width]])
        new_data = scaler.transform(new_data)
        
        # Dự đoán nhãn
        yhat_new = neigh.predict(new_data)
        predicted = label_encoder.inverse_transform(yhat_new)[0]
        print("Dự đoán nhãn:", predicted)
        
        # Hỏi người dùng có tiếp tục không
        if input("Do you want to predict more? (Y or N)").strip().upper() == 'N':
            print("End program")
            break
else:
    print("End program")
