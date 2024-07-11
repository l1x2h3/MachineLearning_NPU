import matplotlib.pyplot as plt  # 导入matplotlib库用于绘图
import pandas as pd
from sklearn.model_selection import train_test_split  # 导入train_test_split函数用于划分数据集
from sklearn.preprocessing import StandardScaler  # 导入StandardScaler类用于数据标准化
from sklearn.neighbors import KNeighborsClassifier  # 导入KNeighborsClassifier类用于创建KNN分类器
from sklearn.metrics import classification_report, confusion_matrix  # 导入分类报告和混淆矩阵计算函数

# 加载鸢尾花数据集
iris = pd.read_csv("iris.csv")
X = iris.drop("Species", axis=1)  # 获取特征数据
y = iris["Species"]  # 获取标签数据

# 数据集分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 划分数据集为80%的训练集和20%的测试集

# 数据标准化
scaler = StandardScaler()  # 创建一个StandardScaler对象
X_train_scaled = scaler.fit_transform(X_train)  # 对训练集进行标准化处理
X_test_scaled = scaler.transform(X_test)  # 对测试集进行标准化处理，使用训练集得到的均值和标准差

# 初始化KNN分类器并训练（使用所有四个特征）
knn = KNeighborsClassifier(n_neighbors=3)  # 创建一个KNN分类器对象，设置邻居数为3
knn.fit(X_train_scaled, y_train)  # 使用训练集数据训练KNN分类器

# 使用测试集进行预测
y_pred = knn.predict(X_test_scaled)  # 使用训练好的KNN分类器对测试集进行预测

# 打印分类报告和混淆矩阵
print(classification_report(y_test, y_pred))  # 打印分类报告，包括精确度、召回率、F1值等
print(confusion_matrix(y_test, y_pred))  # 打印混淆矩阵，展示各类别的分类情况

# 可视化结果（只选择两个特征进行二维可视化）
# 这里我们选择花瓣长度和花瓣宽度作为特征
# 注意：由于数据已经标准化，所以这里的可视化主要是为了展示分类效果，而不是真实的花瓣长度和宽度
plt.scatter(X_test_scaled[y_test == 0, 2], X_test_scaled[y_test == 0, 3], label='Setosa', alpha=0.8)  # 绘制Setosa类别的散点图
plt.scatter(X_test_scaled[y_test == 1, 2], X_test_scaled[y_test == 1, 3], label='Versicolour',
            alpha=0.8)  # 绘制Versicolour类别的散点图
plt.scatter(X_test_scaled[y_test == 2, 2], X_test_scaled[y_test == 2, 3], label='Virginica',
            alpha=0.8)  # 绘制Virginica类别的散点图

# 添加图例和轴标签
plt.xlabel('Petal length (scaled)')  # 这里的'Petal length'是标准化的花瓣长度
plt.ylabel('Petal width (scaled)')  # 这里的'Petal width'是标准化的花瓣宽度
plt.legend()  # 添加图例
plt.show()  # 显示图像

# 保存图像
plt.savefig('knn_iris_visualization.png')  # 将图像保存为'knn_iris_visualization.png'文件
