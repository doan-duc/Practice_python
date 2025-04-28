import pandas as pd
import numpy as np
import pylab as pl
import scipy.optimize as opt
import matplotlib.pyplot as plt

# Load data
churn_df = pd.read_csv(r"C:\Users\DOAN SINH DUC\Downloads\ChurnData.csv")

# Tiền xử lý
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless', 'churn']]
churn_df['churn'] = churn_df['churn'].astype('int')

x = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
y = np.asarray(churn_df['churn'])

#Chuẩn hóa thành các cột có trung bình = 0 và độ lệch chuẩn = 1
from sklearn import preprocessing
x = preprocessing.StandardScaler().fit(x).transform(x)

#Huấn luyện data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= 1)

#Tạo mô hình LR để train
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver = 'liblinear').fit(x_train, y_train)

#Test mô hình
yhat = LR.predict(x_test)
yhat_prob = LR.predict_proba(x_test)

#Đánh giá mô hình
from sklearn.metrics import jaccard_score
jaccard_score(y_test, yhat,pos_label=0)

from sklearn.metrics import classification_report 
print(classification_report(y_test, yhat))

from sklearn.metrics import log_loss
loss = log_loss(y_test, yhat_prob)

print("Loss function = ",loss)
