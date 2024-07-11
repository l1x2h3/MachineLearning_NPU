import pickle
import pandas as pd
import matplotlib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林模型
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import json
from sklearn import tree
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import pydotplus
matplotlib.use('TkAgg')

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

# 使用随机森林模型进行训练
rf_model = RandomForestClassifier()  # 使用默认参数的随机森林模型

accuracies = []
for i in range(1, 100):  # 控制训练轮数
    rf_model.fit(X_train, y_train)  # 在训练集上拟合模型
    y_pred = rf_model.predict(X_test)  # 预测测试集结果
    accuracy = accuracy_score(y_test, y_pred)  # 计算准确率
    accuracies.append(accuracy)  # 将准确率添加到列表中
    if i > 1 and accuracies[-1] == accuracies[-2]:  # 如果准确率连续两次没有提升，则停止绘制
        break

# 绘制准确率随训练轮数变化的曲线图
plt.plot(range(1, len(accuracies) + 1), accuracies)
plt.xlabel('Training Rounds')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Training Rounds (Random Forest)')
plt.show()


# 获取随机森林模型的参数
rf_params = rf_model.get_params()
# 将参数保存到txt文本文件中
with open('random_forest_params.txt', 'w') as file:
    file.write(json.dumps(rf_params, indent=4))  # 将参数以JSON格式写入文件

Estimators = rf_model.estimators_
print(Estimators)
for index, model in enumerate(Estimators):
    filename = 'data_' + str(index) + '.pdf'
    dot_data = tree.export_graphviz(model, out_file=None,
                                    feature_names=data.drop(columns=['y']),
                                    class_names=data['y'],
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(filename)