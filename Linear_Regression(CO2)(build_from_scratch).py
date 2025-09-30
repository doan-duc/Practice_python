import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("FuelConsumption.csv")

#print(df.info())
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

#print(ndf.sample(5))

'''
vẽ biểu đồ phân bố dữ liệuliệu
cdf.hist()
plt.show()
'''
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='yellow', label='ENGINESIZE')
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue', label='FUELCONSUMPTION_COMB')
plt.scatter(cdf.FUELCONSUMPTION_CITY, cdf.CO2EMISSIONS,  color='red', label='FUELCONSUMPTION_CITY')
plt.scatter(cdf.FUELCONSUMPTION_HWY, cdf.CO2EMISSIONS,  color='green', label='FUELCONSUMPTION_HWY')
plt.xlabel('INFLUENCING FACTORS')
plt.ylabel('CO2EMISSIONS')
plt.show()

#Xây dựng module Linear Regression
def LR(X_cols, Y_col):
    X = df[X_cols].values # mxn
    #print(X.shape)
    #print(X.shape[0])
    y = df[Y_col].values.reshape(-1,1)                 # nx1, tạo ma trận cộtcột
    X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
    print(X_bias[0])                                   # Thêm cột intercept hệ số b => n x (m+1) 
    XTX = X_bias.T.dot(X_bias)                         # X^T x X (ma trận vuông)
    XTy = X_bias.T.dot(y)                              # X^T x y
    w = np.linalg.pinv(XTX).dot(XTy)                   # (m+1) x 1 Ma trận hệ số hồi quy (Cần dùng ma trận giả khả nghịch). pinv() dùng được cho ma trận ko khả nghịch
    y_predict = X_bias.dot(w)                          # Dự đoán giá trị y
    return w, y_predict

def MSE(y, y_pred):
    mse = np.mean((y - y_pred) ** 2)
    return mse
def R2(y, y_pred):
    ss_res = np.sum((y - y_pred) ** 2)           
    ss_tot = np.sum((y - np.mean(y)) ** 2)   
    R2 = 1 - (ss_res / ss_tot)
    return R2
influence_factors = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'FUELCONSUMPTION_CITY']
Co2 = ['CO2EMISSIONS']
y = df[Co2].values
w, y_predict = LR(influence_factors, Co2)
mse = MSE(y, y_predict)
R2 = R2(y, y_predict)
print("Trong so w = ", w)
print(f"MSE: {mse:.2f}")
print(f"R^2: {R2:.2f}")
plt.scatter(y, y_predict)
plt.xlabel("Thực tế (y)")
plt.ylabel("Dự đoán (ŷ)")
plt.title("So sánh y thực và y dự đoán")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r-')  # đường y = ŷ
plt.grid(True)
plt.show()
