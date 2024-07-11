import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data_dir = './cifar-10-batches-py/'  # 解压后的数据集目录


# 加载数据
def load_data(file):
    import pickle
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


def load_cifar10_data(data_dir):
    train_x, train_y = [], []
    for i in range(1, 6):
        filename = os.path.join(data_dir, f'data_batch_{i}')
        data_dict = load_data(filename)
        train_x.extend(data_dict[b'data'])
        train_y.extend(data_dict[b'labels'])

    test_data_dict = load_data(os.path.join(data_dir, 'test_batch'))
    test_x, test_y = test_data_dict[b'data'], test_data_dict[b'labels']

    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)


train_x, train_y, test_x, test_y = load_cifar10_data(data_dir)
print('\n train_x:%s, train_y:%s, test_x:%s, test_y:%s' % (train_x.shape, train_y.shape, test_x.shape, test_y.shape))

# 数据预处理
X_train, X_test = train_x / 255.0, test_x / 255.0
y_train, y_test = train_y.flatten(), test_y.flatten()

# 初始化SVM模型
svm = SVC(kernel='linear', verbose=2)

# 训练SVM模型
svm.fit(X_train, y_train)

# 预测测试集数据
y_pred = svm.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print('SVM模型在测试集上的准确率：', accuracy)
