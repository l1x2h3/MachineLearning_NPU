import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 读取数据集
data = pd.read_csv('bank.csv')

# 对2，3，4，5字段的单词类别进行整数编码处理
label_enc = LabelEncoder()
data['job'] = label_enc.fit_transform(data['job'])
data['marital'] = label_enc.fit_transform(data['marital'])
data['education'] = label_enc.fit_transform(data['education'])
data['default'] = label_enc.fit_transform(data['default'])
data['housing'] = label_enc.fit_transform(data['housing'])
data['loan'] = label_enc.fit_transform(data['loan'])
data['contact'] = label_enc.fit_transform(data['contact'])
data['month'] = label_enc.fit_transform(data['month'])
data['poutcome'] = label_enc.fit_transform(data['poutcome'])
data['y'] = label_enc.fit_transform(data['y'])

# 折线统计图：年龄
plt.figure(figsize=(10, 6))
data['age'].value_counts().sort_index().plot(kind='line')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.grid()
plt.show()

# 饼图统计图：职业
plt.figure(figsize=(8, 8))
data['job'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Job')
plt.ylabel('')
plt.show()