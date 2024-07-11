import pandas as pd
import matplotlib
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC  # 导入SVM模型
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# 读取数据集
data = pd.read_csv('bank-full.csv')

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 使用SVM模型进行训练
svm_model = SVC()  # 使用默认参数的SVM模型

accuracies = []
for i in range(1, 100):  # 控制训练轮数
    svm_model.fit(X_train, y_train)  # 在训练集上拟合模型
    y_pred = svm_model.predict(X_test)  # 预测测试集结果
    accuracy = accuracy_score(y_test, y_pred)  # 计算准确率
    accuracies.append(accuracy)  # 将准确率添加到列表中
    if i > 31:
        # and accuracies[-1] == accuracies[-2]:  # 如果准确率连续两次没有提升，则停止绘制
        break

# 绘制准确率随训练轮数变化的曲线图
plt.plot(range(1, len(accuracies) + 1), accuracies)
plt.xlabel('Training Rounds')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Training Rounds (SVM)')
plt.show()