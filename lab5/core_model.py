import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import os

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

X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

# 其他部分的代码可直接使用之前提供的修改后代码

# 建立模型
model = tf.keras.Sequential()
##特征提取阶段
# 第一层
model.add(
    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, data_format='channels_last',
                           input_shape=X_train.shape[1:]))  # 卷积层，16个卷积核，大小（3，3），保持原图像大小，relu激活函数，输入形状（28，28，1）
model.add(tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),padding='same'))  # 池化层，最大值池化，卷积核（2，2）
# 第二层
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu))
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),padding='same'))
##分类识别阶段
# 第三层
model.add(tf.keras.layers.Flatten())  # 改变输入形状
# 第四层
model.add(tf.keras.layers.Dense(128, activation='relu'))  # 全连接网络层，128个神经元，relu激活函数
model.add(tf.keras.layers.Dense(10, activation='softmax'))  # 输出层，10个节点
print(model.summary())  # 查看网络结构和参数信息

# 配置模型训练方法
# adam算法参数采用keras默认的公开参数，损失函数采用稀疏交叉熵损失函数，准确率采用稀疏分类准确率函数
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

# 训练模型
# 批量训练大小为64，迭代5次，测试集比例0.2（48000条训练集数据，12000条测试集数据）
print('--------------')
nowtime = time.strftime('%Y-%m-%d %H:%M:%S')
print('训练前时刻：' + str(nowtime))

history = model.fit(X_train, y_train, batch_size=256, epochs=1, validation_split=0.5)

print('--------------')
nowtime = time.strftime('%Y-%m-%d %H:%M:%S')
print('训练后时刻：' + str(nowtime))

# 评估模型
model.evaluate(X_test, y_test, verbose=2)  # 每次迭代输出一条记录，来评价该模型是否有比较好的泛化能力

# 保存整个模型
model.save('CIFAR10_CNN_weights.h5')

# 结果可视化
print(history.history)
loss = history.history['loss']  # 训练集损失
val_loss = history.history['val_loss']  # 测试集损失
acc = history.history['sparse_categorical_accuracy']  # 训练集准确率
val_acc = history.history['val_sparse_categorical_accuracy']  # 测试集准确率

plt.figure(figsize=(10, 3))

plt.subplot(121)
plt.plot(loss, color='b', label='train')
plt.plot(val_loss, color='r', label='test')
plt.ylabel('loss')
plt.legend()

plt.subplot(122)
plt.plot(acc, color='b', label='train')
plt.plot(val_acc, color='r', label='test')
plt.ylabel('Accuracy')
plt.legend()

# 暂停5秒关闭画布，否则画布一直打开的同时，会持续占用GPU内存
# 根据需要自行选择
plt.ion()       #打开交互式操作模式
plt.show()


# 使用模型
plt.figure()
for i in range(10):
    num = np.random.randint(1, 10000)

    plt.subplot(2, 5, i + 1)
    plt.axis('off')
    image = test_x[num]
    #plt.imshow(image, cmap='gray')
    demo = tf.reshape(X_test[num], (1, 32, 32, 3))
    # y_pred = np.argmax(model.predict(demo))
    y_pred = np.argmax(model.predict(X_test[0:5]), axis=1)
    plt.title('标签值：' + str(test_y[num]) + '\n预测值：' + str(y_pred))

    # print('X_test[0:5]: %s'%(X_test[0:5].shape))
    # print('y_pred: %s'%(y_pred))

plt.ion()       #打开交互式操作模式
plt.show()

while True:
    plt.pause(0.1)
    time.sleep(5)