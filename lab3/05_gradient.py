import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json

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

# 将预测结果字段'y'提取出来
X = data.drop(columns=['y'])
y = data['y']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# 使用梯度提升法进行训练
gb_model = GradientBoostingClassifier()  # 使用默认参数的梯度提升法模型
gb_model.fit(X_train, y_train)  # 在训练集上拟合模型
y_pred = gb_model.predict(X_test)  # 预测测试集结果

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 绘制准确率随训练轮数变化的曲线图
plt.plot(range(1, gb_model.n_estimators + 1), gb_model.train_score_)
plt.xlabel('Training Rounds')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Training Rounds (Gradient Boosting)')
plt.show()

# 获取梯度提升法模型的参数
gb_params = gb_model.get_params()

# 将参数保存到txt文本文件中
with open('gradient_boosting_params.txt', 'w') as file:
    file.write(json.dumps(gb_params, indent=4))  # 将参数以JSON格式写入文件
