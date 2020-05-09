# 1. 建立多元回归模型——波士顿房价预测
## 数据集
[下载链接](https://aistudio.baidu.com/aistudio/datasetdetail/5646)

|属性|含义|属性|含义|
|--|--|--|--|
CRIM|城镇人均犯罪率|ZN|住宅用地超过 25000 sq.ft. 的比例。
INDUS|城镇非零售商用土地的比例|CHAS|查理斯河空变量（如果边界是河流，则为1；否则为0）
NOX|一氧化氮浓度|RM|住宅平均房间数|AGE|1940 年之前建成的自用房屋比例
|DIS|到波士顿五个中心区域的加权距离|RAD|辐射性公路的接近指数
TAX|每 10000 美元的全值财产税率|PTRATIO|城镇师生比例
B|1000（Bk-0.63）^ 2，其中 Bk 指代城镇中黑人的比例|LSTAT|人口中地位低下者的比例。
MEDV|自住房的平均房价，以千美元计

## 使用的第三方库
```py
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
```
## 读取并处理数据
- 读取数据
```py
data = pd.read_csv('data/housing.csv')
```
- 不考虑城镇人均犯罪率，模型评分较高 
   [iloc用法](https://blog.csdn.net/qq_39753778/article/details/105940039)
```py
# 不要第一列的数据
new_data = data.iloc[:, 1:]
```
## 查看数据
- 查看处理后的数据集
```py
# 得到数据集且查看
print('head:', new_data.head(), '\nShape:', new_data.shape)
```
- 检查是否存在缺失值
```py
# 缺失值检验
print(new_data.isnull().sum())
```

## 查看数据分散情况——绘制箱形图
- 输出`行数count`，`平均值mean`，`标准差std`，`最小值min`，`最大值max`，`上四分位数75%`, `中位数50%`，`下四分位数25%`
```py
print(new_data.describe())
```
- 箱形图（Box-plot）是一种用作显示一组数据分散情况资料的统计图。

- 箱线图的绘制方法是：先找出一组数据的上边缘、下边缘、中位数和两个四分位数；然后， 连接两个四分位数画出箱体；再将上边缘和下边缘与箱体相连接，中位数在箱体中间。
<img src="https://img-blog.csdnimg.cn/20200505224214767.png" width=70%>

- 箱型图绘制代码
```py
new_data.boxplot()
plt.show()
```
<img src="https://img-blog.csdnimg.cn/20200506005827595.png" width=70%>

## 数据集分割
将原始数据按照`2:8`比例分割为“测试集”和“训练集”
```py
X_train, X_test, Y_train, Y_test = train_test_split(new_data.iloc[:, :13], new_data.MEDV, train_size=.80)
```
## 建立多元回归模型
根据训练集建立模型
```py
model = LinearRegression()
model.fit(X_train, Y_train)
a = model.intercept_
b = model.coef_
print("最佳拟合线:截距", a, ",回归系数：", b)

score = model.score(X_test, Y_test)
print(score)
```
```
最佳拟合线:截距 0.0 ,回归系数： [-1.74325842e-16  1.11629233e-16 -1.79794258e-15  7.04652389e-15
 -2.92277767e-15  2.97853711e-17 -8.23334194e-16  1.17159575e-16
  1.88696229e-17 -3.41643920e-16 -1.28401929e-17 -5.78208730e-17
  1.00000000e+00]
1.0
```
## 测试
```py
Y_pred = model.predict(X_test)
print(Y_pred)
plt.plot(range(len(Y_pred)), Y_pred, 'b', label="predict")
plt.show()
```
### 画图表示结果
```py
X_train, X_test, Y_train, Y_test = train_test_split(new_data.iloc[:, :13], new_data.MEDV, train_size=.80)

plt.figure()
plt.plot(range(len(Y_pred)), Y_pred, 'b', label="predict")
plt.plot(range(len(X_test)), Y_test, 'r', label="test")
plt.legend(loc="upper right")
plt.xlabel("the number of MEDV")
plt.ylabel('value of MEDV')
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200506014618510.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NzUzNzc4,size_16,color_FFFFFF,t_70)
