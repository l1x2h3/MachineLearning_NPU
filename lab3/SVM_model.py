import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=78)
# 使用SVM模型进行训练
svm_model = SVC()
svm_model.fit(X_train, y_train)
# 预测测试集结果
y_pred = svm_model.predict(X_test)
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of SVM model: ', accuracy)