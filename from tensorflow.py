# Import các thư viện cần thiết
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Dense #type: ignore
from tensorflow.keras.utils import to_categorical#type: ignore

# Tải bộ dữ liệu
df = pd.read_csv(r"C:\Users\DOAN SINH DUC\Downloads\archive\IRIS.csv")

# Tạo dataframe và chuẩn hóa
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

# Mã hóa nhãn bằng LabelEncoder và chuyển thành one-hot vector
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded)

# Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.3, random_state=4, stratify=y)

# Chuẩn hóa dữ liệu đầu vào
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Xây dựng mô hình NN
model = Sequential([
    Dense(128, activation='relu', input_shape=(4,)),  # Lớp đầu vào với 4 đặc trưng
    Dense(64, activation='relu'),                     # Lớp ẩn
    Dense(3, activation='softmax')                    # Lớp đầu ra với 3 lớp (Iris-setosa, Iris-versicolor, Iris-virginica)
])

# Compile mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.2, verbose=1)

# Đánh giá mô hình
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Dự đoán trên tập test
yhat = model.predict(X_test)
yhat_classes = np.argmax(yhat, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Báo cáo đánh giá
print(classification_report(y_test_classes, yhat_classes, target_names=label_encoder.classes_))

# Dự đoán thử một mẫu nhập từ bàn phím
print("Do you want to predict one? (Y or N)")
if input().strip().upper() == 'Y':
    while True:
        # Nhập giá trị từ bàn phím
        sepal_length = float(input("Enter sepal_length: "))
        sepal_width = float(input("Enter sepal_width: "))
        petal_length = float(input("Enter petal_length: "))
        petal_width = float(input("Enter petal_width: "))
        
        # Xây dựng dữ liệu mới và chuẩn hóa
        new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        new_data = scaler.transform(new_data)
        
        # Dự đoán nhãn
        yhat_new = model.predict(new_data)
        predicted_class = np.argmax(yhat_new, axis=1)[0]
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
        print("Dự đoán nhãn:", predicted_label)
        
        # Hỏi người dùng có tiếp tục không
        if input("Do you want to predict more? (Y or N)").strip().upper() == 'N':
            print("End program")
            break
else:
    print("End program")
