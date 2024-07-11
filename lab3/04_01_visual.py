# 导入必要的库
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # 绘图库
from sklearn.preprocessing import LabelEncoder
# 解决画图中文字体显示的问题
plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman']  # 汉字字体集
plt.rcParams['font.size'] = 10  # 字体大小
plt.rcParams['axes.unicode_minus'] = False
# 忽略警告


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
# 定义一个随机森林回归模型
RF = RandomForestRegressor(n_jobs=-1)
# 训练模型
RF.fit(X, y)
# 获取特征重要性得分
feature_importances = RF.feature_importances_
# 创建特征名列表
feature_names = list(X.columns)
# 创建一个DataFrame，包含特征名和其重要性得分
feature_importances_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
# 对特征重要性得分进行排序
feature_importances_df = feature_importances_df.sort_values('importance', ascending=False)

# 颜色映射
colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))

# 可视化特征重要性
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(feature_importances_df['feature'], feature_importances_df['importance'], color=colors)
ax.invert_yaxis()  # 翻转y轴，使得最大的特征在最上面
ax.set_xlabel('特征重要性', fontsize=12)  # 图形的x标签
ax.set_title('随机森林特征重要性可视化', fontsize=16)
for i, v in enumerate(feature_importances_df['importance']):
    ax.text(v + 0.01, i, str(round(v, 3)), va='center', fontname='Times New Roman', fontsize=10)

# # 设置图形样式
ax.spines['top'].set_visible(False)  # 去掉上边框
ax.spines['right'].set_visible(False)  # 去掉右边框

# 保存图形
plt.savefig('./特征重要性.jpg', dpi=400, bbox_inches='tight')
plt.show()