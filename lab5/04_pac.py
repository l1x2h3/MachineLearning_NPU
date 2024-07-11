import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 加载数据
data_dir = './cifar-10-batches-py/'

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

# 数据预处理
X_train = train_x / 255.0
y_train = train_y.flatten()

# 将数据降维到二维
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

# 绘制数据可视化图
plt.figure(figsize=(10, 8))
colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'black', 'brown', 'gray']
for i in range(10):
    indices = np.where(y_train == i)
    plt.scatter(X_train_pca[indices, 0], X_train_pca[indices, 1], label=f'Label {i}', color=colors[i])

plt.title('CIFAR-10 Dataset Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='best')
plt.show()