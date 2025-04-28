import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# đọc file FuelConsumption bằng câu lệnh read_csv() trong thư viện pandas
df = pd.read_csv("FuelConsumption.csv")

# in ra các dòng của mẫu dữ liệu, nếu trong head() rỗng thì mặc định là 5. Nếu muốn in số dòng chỉ định thì dùng print(df.iloc[a:b])
print(df.head())#df viết tắt cho dataframe

# tạo 1 list mới gồm những thông tin của các cột có tên ở dưới
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]


print(cdf.head(9))

# Dùng thư viện plt đề vẽ biểu đồ
# plt.scatter(a, b, color='x') trong đó a, b tương ứng với x, y là các nội dung thông tin trong biểu đồ
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='red')
# plt.xlabel và plt.ylabel dùng để đặt tên cho cột x và y
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


