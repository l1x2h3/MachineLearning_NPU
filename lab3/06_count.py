import pandas as pd
import matplotlib.pyplot as plt

# 读取数据集
data = pd.read_csv('bank.csv')

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