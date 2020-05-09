import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# 读取波士顿房价数据集（数据集下载链接：https://aistudio.baidu.com/aistudio/datasetdetail/5646）
data = pd.read_csv('data/housing.csv')
# 不考虑城镇人均犯罪率，模型评分较高
new_data = data.iloc[:, 1:]
# 得到数据集且查看
print('head:', new_data.head(), '\nShape:', new_data.shape)

print(new_data.describe())
# 缺失值检验
print(new_data[new_data.isnull() == True].count())
new_data.boxplot()
plt.show()
print(data.corr())
print(new_data.corr())
X_train, X_test, Y_train, Y_test = train_test_split(new_data.iloc[:, :13], new_data.MEDV, train_size=.80)
print("原始数据特征:", new_data.iloc[:, :13].shape, ",训练数据特征:", X_train.shape, ",测试数据特征:", X_test.shape)
print("原始数据标签:", new_data.MEDV.shape, ",训练数据标签:", Y_train.shape, ",测试数据标签:", Y_test.shape)

model = LinearRegression()
model.fit(X_train, Y_train)
a = model.intercept_
b = model.coef_
print("最佳拟合线:截距", a, ",回归系数：", b)
# 模型评分
score = model.score(X_test, Y_test)
print(score)

Y_pred = model.predict(X_test)
print(Y_pred)
plt.plot(range(len(Y_pred)), Y_pred, 'b', label="predict")
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(new_data.iloc[:, :13], new_data.MEDV, train_size=.80)

plt.figure()
plt.plot(range(len(Y_pred)), Y_pred, 'b', label="predict")
plt.plot(range(len(X_test)), Y_test, 'r', label="test")
plt.legend(loc="upper right")
plt.xlabel("the number of MEDV")
plt.ylabel('value of MEDV')
plt.show()
