import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import io
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import radians, sin, cos, sqrt, atan2
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import os


data_path = "/Users/liuzhenke/PycharmProjects/baseline st prediction/point_processed"

#print(f"Checking path: {data_path}")

csv_file_path = 'space-time-lr.csv'

from scipy.spatial.distance import cdist

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline  # 添加了Pipeline的导入

# 读取训练数据
train_data = pd.read_csv('train_data.csv')

# 提取特征和目标变量
X_train = train_data[['date', 'lat', 'lon']]
y_train = train_data['temperture']

# 初始化OneHotEncoder，用于处理日期的编码
date_encoder = OneHotEncoder()

# 列变换器，将日期进行独热编码
preprocessor = ColumnTransformer(
    transformers=[
        ('date', date_encoder, ['date'])
    ],
    remainder='passthrough'
)

# 初始化线性回归模型，使用列变换器进行预处理
model = LinearRegression()

# 构建管道，包括预处理和模型
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# 训练模型
pipeline.fit(X_train, y_train)

# 读取测试数据
test_data = pd.read_csv('test_data.csv')

# 提取测试数据的特征
X_test = test_data[['date', 'lat', 'lon']]

# 在测试数据上进行预测
y_pred = pipeline.predict(X_test)

# 将预测结果与实际温度进行比较
test_data['predicted_temp'] = y_pred

# 打印前几行测试数据及预测结果
print(test_data[['date', 'lat', 'lon', 'temperture', 'predicted_temp']])

test_data[['date', 'lat', 'lon', 'temperture', 'predicted_temp']].to_csv('space-time-lr.csv')
df = pd.read_csv("space-time-lr.csv")
prediction_temp = df['predicted_temp']
actual_temp = df['temperture']


mae = mean_absolute_error(actual_temp, prediction_temp)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual_temp, prediction_temp))

# Calculate R²
r2 = r2_score(actual_temp, prediction_temp)

df['MAE'] = mae
df['RMSE'] = rmse
df['R²'] = r2
df.to_csv("space-time-lr.csv", index=False)

print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)

# Sample train_data and test_data
# Make sure to replace this with your actual train_data and test_data
'''
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')
print(train_data)
print(test_data)
train_data['date'] = pd.to_datetime(train_data['date']).astype(int)

# 将数据分为特征（X）和目标变量（y）
X_train = train_data[['lat', 'lon', 'date']]
y_train = train_data['temperture']

# 创建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 对测试数据进行预测
# 假设'test_data'的'date'列也需要转换为数值格式
test_data['date'] = pd.to_datetime(test_data['date']).astype(int)
X_test = test_data[['lat', 'lon', 'date']]
predictions = model.predict(X_test)

test_data['predicted_temperature'] = predictions
test_data['date'] = pd.to_datetime(test_data['date'])


# 可以根据需要使用预测值了
print(test_data)
test_data.to_csv("space-time-lr.csv")
df = pd.read_csv("space-time-lr.csv")
prediction_temp = df['predicted_temperature']
actual_temp = df['temperture']


mae = mean_absolute_error(actual_temp, prediction_temp)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual_temp, prediction_temp))

# Calculate R²
r2 = r2_score(actual_temp, prediction_temp)

df['MAE'] = mae
df['RMSE'] = rmse
df['R²'] = r2
df.to_csv("space-time-lr.csv", index=False)

print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)
'''
'''
print(test_data)
test_data[['date', 'lat', 'lon', 'temperture', 'predicted_temperature']].to_csv("space-time-nn.csv")
df = pd.read_csv("space-time-nn.csv")
prediction_temp = df['predicted_temperature']
actual_temp = df['temperture']

mae = mean_absolute_error(actual_temp, prediction_temp)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual_temp, prediction_temp))

# Calculate R²
r2 = r2_score(actual_temp, prediction_temp)

df['MAE'] = mae
df['RMSE'] = rmse
df['R²'] = r2
df.to_csv("space-time-lr.csv", index=False)

print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)
'''
