import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# 假设data是您的数据集
data = pd.read_csv('bank.csv')
# 按poutcome统计每个段的人数
poutcome_count = data['poutcome'].value_counts()
print("按poutcome统计每个段的人数:")
print(poutcome_count)

# 对y进行分类统计成功率是yes还是no
y_count = data['y'].value_counts()
y_success_rate = data[data['y'] == 1]['y'].count() / data['y'].count()
print("\n对y进行分类统计成功率:")
print("Yes的数量:", y_count[1])
print("No的数量:", y_count[0])
print("成功率 (Yes):", y_success_rate)

# 用扇形统计图统计结婚和未婚的人数
marital_status_count = data['marital'].value_counts()
plt.figure(figsize=(8, 8))
marital_status_count.plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Marital Status')
plt.ylabel('')
plt.show()



# 假设 data 是包含 'poutcome' 数据的 Pandas Series 或 Numpy 数组
plt.figure(figsize=(8, 6))
sns.kdeplot(data, shade=True, color='blue')
plt.xlabel('poutcome')
plt.ylabel('Density')
plt.title('Density Plot of poutcome')
plt.show()