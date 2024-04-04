import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import polynomial_regression_program as poly


# 构建数据集
m = 100
X = 6*np.random.rand(m,1) - 3
y = 0.5*X**2+X+np.random.randn(m,1)

################################################################
# # 将训练数据进行可视化
# plt.plot(X,y,'b.')
# plt.xlabel('X_1')
# plt.ylabel('y')
# plt.axis([-3,3,-5,10])
# plt.show()

#################################################################
## 任务二：观察过拟合与欠拟合的图像

# # 观察不同度数下的拟合效果
# X_new = np.linspace(-3,3,100).reshape(100,1)
# plt.title('Underfitting, Overfitting, and Standard Fitting')

# # 欠拟合
# poly.polynomial_regression_fit(X, y, X_new, degree = 1)
# # 标准拟合
# poly.polynomial_regression_fit(X, y, X_new, degree = 2)
# # 过拟合
# poly.polynomial_regression_fit(X, y, X_new, degree = 100)
# # 展示图像
# plt.show()

##################################################################
## 观察样本数量对结果的影响

# lin_reg = LinearRegression()
# poly.plot_learning_curves(lin_reg,X,y)
# plt.axis([0,80,0,3.3])
# plt.show()

##################################################################
## 任务三：多项式的过拟合

np.random.seed(31)

# 使用多项式特征的线性回归模型管道 polynomial_reg
polynomial_reg = Pipeline([
    ('poly_features',PolynomialFeatures(degree = 25,include_bias= False)), 
    ('lin_reg',LinearRegression()
     )])

poly.plot_learning_curves(polynomial_reg,X,y)
plt.axis([0,80,0,5])
plt.show()