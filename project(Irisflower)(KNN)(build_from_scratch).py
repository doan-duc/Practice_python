# Import các thư viện cần thiết
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Hàm chuẩn hóa dữ liệu
def standardscaler (x): 
    return (x-np.mean(x, axis = 0))/x.std(axis = 0) #(x-x trung bình)/phương sai

#Hàm tính khoảng cách
def distance(point1, point2):
    return np.sqrt(np.sum((point1-point2)**2))
#Hàm tìm k điểm lân cận
def find_k_neighbors(X_train, y_train, point, k):
    d = [distance(point, x) for x in X_train]
    k_neighbors = np.argsort(d)[:k]#mảng gồm các chỉ số của k điểm có khoảng cách gần nhất
    k_label = y_train[k_neighbors]#mảng gồm các nhãn của k điểm có khoảng cách gần nhất
    return k_label

#Hàm chia dữ liệu
def split_data(X, y, test_size, random_state):
    amount = X.shape[0]
    test_amount = int(amount * test_size)
    np.random.seed(random_state)
    index_array = np.arange(amount)
    np.random.shuffle(index_array)#Đảo vị trí các phần tử trong mảng ko tạo mảng mới
    X_test, X_train = X[index_array[:test_amount]], X[index_array[test_amount:]]
    y_test, y_train = y[index_array[:test_amount]], y[index_array[test_amount:]]
    return X_train, X_test, y_train, y_test

#Hàm dự đoán
def predict1(X_train, y_train, point, k):
    a = find_k_neighbors(X_train, y_train, point, k)
    return round(np.mean(a, axis = 0))

def predict(X_train, y_train, X_test, k):
    a = np.array([predict1(X_train, y_train, point, k)for point in X_test])
    return a

# Đánh giá mô hình
def accuracy(result, y_test):
    count = 0
    for i in range(len(result)):
        if result[i] == y_test[i]:
            count += 1
    return count/len(result)
# Tải bộ dữ liệu
df = pd.read_csv(r"C:\Users\DOAN SINH DUC\Downloads\archive\IRIS.csv")

# Tạo dữ liệu
X = np.asarray(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
y = np.asarray(df['species'])
encoder = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y_encoder = np.array(list(map(encoder.get, y)))
#Vẽ biểu đồ
x_feature = X[:, 2]
y_feature = X[:, 3]
colors = ['red', 'blue', 'green']
for i in range(3):
    plt.scatter(x_feature[y_encoder == i], y_feature[y_encoder == i], color = colors[i])
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Scatter Plot - Petal Length vs Petal Width')
plt.show()
#Chuẩn hóa và xây dựng mô hình
X_train, X_test, y_train, y_test = split_data(X, y_encoder, test_size = 0.3, random_state = 42)

X_train = standardscaler(X_train)
X_test = standardscaler(X_test)

result_test = predict(X_train, y_train, X_test, k = 3)
print("Test set accuracy: ", accuracy(result_test, y_test))
result_train = predict(X_train, y_train, X_train, k = 3)
print("Train set Accuracy: ", accuracy(result_train, y_train))
