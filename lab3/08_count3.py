import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('bank.csv')
# 统计月份数量并绘制柱状图
plt.figure(figsize=(8, 6))
data['month'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.xlabel('Month')
plt.ylabel('Count')
plt.title('Count of Occurrences by Month')
plt.xticks(rotation=0)
plt.show()

# 对 duration 进行区间分布统计并绘制柱状图
duration_bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
duration_labels = ['0-100', '101-200', '201-300', '301-400', '401-500', '501-600', '601-700', '701-800', '801-900', '901-1000']

data['duration_interval'] = pd.cut(data['duration'], bins=duration_bins, labels=duration_labels)
plt.figure(figsize=(8, 6))
data['duration_interval'].value_counts().sort_index().plot(kind='bar', color='lightcoral')
plt.xlabel('Duration Interval')
plt.ylabel('Count')
plt.title('Count of Occurrences by Duration Interval')
plt.xticks(rotation=45)
plt.show()