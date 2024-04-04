import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import warnings
import linear_regression_program as lr
from sklearn.linear_model import LinearRegression

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

warnings.filterwarnings('ignore')
np.random.seed(42)

# 准备数据集
X = 2*np.random.rand(100,1)
y = 4+ 3*X +np.random.randn(100,1)

# 可视化展示数据集
# plt.plot(X,y,'b.')
# plt.xlabel('X_1')
# plt.ylabel('y')
# plt.axis([0,2,0,15])
# plt.show()

# 最小二乘法直接计算线性回归模型的参数
def least_squares_linear(X, y):
    print('最小二乘法进行线性拟合：')
    theta_best = lr.least_squares_linear_fit(X, y)
    print('theta_best = ')
    print(theta_best)

    # 基于之前计算的theta对数据进行预测
    X_new = np.array([[0],[2]])
    # 这里增加一列是为了对应回归模型中的截距项
    X_new_b = np.c_[np.ones((2,1)),X_new]
    # 这里是theta*x
    y_predict = lr.least_squares_linear_predict(theta_best, X_new_b)
    print('y_predict = ')
    print(y_predict)

    # 可视化展示拟合结果
    lr.chart(X_new, y_predict, X, y, 'least_squares_linear')

def gradient_descent_linear(X, y):
    print('普通梯度下降法进行线性拟合：')

    # 这里对模型进行拟合
    lin_reg = lr.gradient_descent_linear_fit(X, y)
    # 这里输出拟合的参数
    # 输出截距
    print('lin_reg.intercept = ')
    print (lin_reg.intercept_) 
    # 输出系数
    print('lin_reg.coef = ')
    print (lin_reg.coef_) 

    # 基于之前的训练参数对数据进行预测
    X_new = np.array([[0],[2]])
    y_predict = lr.gradient_descent_linear_predict(lin_reg, X_new)
    print('y_predict = ')
    print(y_predict)

    # 可视化展示拟合结果
    lr.chart(X_new, lin_reg.predict(X_new), X, y, 'gradient_descent_linear')


def batch_gradient_descent_linear(X, y):
    print('使用批梯度下降法进行线性拟合：')
    # 设置学习率为0.1
    eta = 0.1
    # # 设置迭代次数为1000次
    # n_iterations = 1000
    # # 假设我们有100个样本
    # m = 100

    # 随机初始化参数theta
    theta = np.random.randn(2,1)

    # gradients, theta = lr.batch_gradient_descent_linear_fit(n_iterations, X, y, m, theta, eta)
    theta_path_bgd = []
    # 基于之前的参数对数据进行预测
    X_new = np.array([[0],[2]])
    # 这里增加一列是为了对应回归模型中的截距项
    X_new_b = np.c_[np.ones((2,1)),X_new]

    # plt.figure(figsize=(10,4))
    # plt.subplot(131)
    # lr.plot_gradient_descent(X, y, X_new, X_new_b, theta, eta = 0.02)
    # plt.subplot(132)
    # lr.plot_gradient_descent(X, y, X_new, X_new_b, theta, eta = 0.1)
    # plt.subplot(133)
    # lr.plot_gradient_descent(X, y, X_new, X_new_b, theta, eta = 0.5)
    theta = lr.plot_gradient_descent(X, y, X_new, X_new_b, theta, eta, theta_path_bgd)
    print('theta = ')
    print(theta)
    plt.show()

    return theta_path_bgd

def stochastic_gradient_descent_linear(X, y):
    print('使用随机梯度下降法进行线性拟合：')
    # 基于之前计算的theta对数据进行预测
    X_new = np.array([[0],[2]])
    X_b = np.c_[np.ones((100,1)),X]
    # 这里增加一列是为了对应回归模型中的截距项
    X_new_b = np.c_[np.ones((2,1)),X_new]

    # 存储优化的 theta 值
    theta_path_sgd=[]

    # 读取样本长度
    m = len(X_b)

    # 设置随机数种子
    np.random.seed(42)

    # 设置遍历整个数据集的次数
    n_epochs = 50

    # 学习调度的参数，用于逐渐减小学习率
    t0 = 5
    t1 = 50

    # 开始训练样本，同时作图可视化
    theta = lr.stochastic_gradient_descent_linear_fit(n_epochs, m, X, y, X_new, t0, t1, theta_path_sgd)
    plt.plot(X,y,'b.')
    plt.axis([0,2,0,15])
    plt.show()
    print('theta = ')
    print(theta)

    return theta_path_sgd

def miniBatch_gradient_descent_linear(X, y):
    print('使用小批量梯度下降法进行线性拟合:')
    # 记录更新的 theta 值
    theta_path_mgd=[]
    # 设置更新参数 theta 的次数
    n_epochs = 100
    # 设置单次循环训练的数据样本个数
    minibatch = 16
    # 初始化 theta 值
    theta = np.random.randn(2,1)
    # 调节学习率的参数，用于逐渐减小学习率
    t0, t1 = 200, 1000
    t = 0

    np.random.seed(42)
    X_new = np.array([[0],[2]])
    theta = lr.miniBatch_gradient_descent_linear_fit(n_epochs, minibatch, theta, X, y, X_new, t, t0, t1, theta_path_mgd)

    plt.show()
    print('theta = ')
    print(theta)

    return theta_path_mgd

def gradient_descent_path_comparison(X, y):
    theta_path_bgd = np.array(batch_gradient_descent_linear(X, y))
    theta_path_sgd = np.array(stochastic_gradient_descent_linear(X, y))
    theta_path_mgd = np.array(miniBatch_gradient_descent_linear(X, y))

    plt.figure(figsize=(12,6))
    plt.plot(theta_path_sgd[:,0],theta_path_sgd[:,1],'r-s',linewidth=1,label='SGD')
    plt.plot(theta_path_mgd[:,0],theta_path_mgd[:,1],'g-+',linewidth=2,label='MINIGD')
    plt.plot(theta_path_bgd[:,0],theta_path_bgd[:,1],'b-o',linewidth=3,label='BGD')
    plt.legend(loc='upper left')
    plt.axis([3.5,4.5,2.0,4.0])
    plt.show()

# least_squares_linear(X, y)
# gradient_descent_linear(X, y)
# batch_gradient_descent_linear(X, y)
# stochastic_gradient_descent_linear(X, y)
# miniBatch_gradient_descent_linear(X, y)
gradient_descent_path_comparison(X, y)
