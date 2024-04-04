import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LinearRegression

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

warnings.filterwarnings('ignore')
np.random.seed(42)

def least_squares_linear_fit(X, y):
    X_b = np.c_[np.ones((100,1)),X]
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta_best

def least_squares_linear_predict(theta_best, X_new_b):
    # 这里是theta*x
    y_predict = X_new_b.dot(theta_best)
    return y_predict

def chart(X_new=None, y_predict=None, X=None, y=None, title=None):
    # 可视化展示拟合结果
    plt.title(title)
    plt.plot(X_new, y_predict, 'r--')
    plt.plot(X, y, 'b.')
    plt.axis([0,2,0,15])
    plt.show()

def gradient_descent_linear_fit(X, y):
    lin_reg = LinearRegression()
    lin_reg.fit(X,y)
    return lin_reg

def gradient_descent_linear_predict(lin_reg, X):
    return lin_reg.predict(X)

def plot_gradient_descent(X, y, X_new, X_new_b, theta, eta, theta_path_bgd):

    # 增加了一列1，代表的是截距项
    X_b = np.c_[np.ones((100,1)),X]
    m = len(X_b)

    # 先作出原始图像
    plt.plot(X, y,'b.')

    # 设置遍历数据的次数
    n_iterations = 1000
    for iteration in range(n_iterations):
        # 进行一次预测，并且作图
        y_predict = X_new_b.dot(theta)
        plt.plot(X_new,y_predict,'r-')

        # 计算样本的梯度
        gradients = 2/m * X_b.T.dot(X_b.dot(theta)-y)

        # 更新模型的参数 theta
        theta = theta - eta*gradients
        if theta_path_bgd is not None:
            theta_path_bgd.append(theta)
        plt.xlabel('X_1')
        plt.axis([0,2,0,15])
        plt.title('batch_gradient_descent_linear(eta = {})'.format(eta))
    return theta

def batch_gradient_descent_linear_fit(n_iterations, X, y, m, theta, eta):
    for iteration in range(n_iterations):
        gradients = 2/m* X.T.dot(X.dot(theta)-y)
        theta = theta - eta*gradients
    return gradients, theta

def learning_schedule(t0, t1, t):
    # 计算学习率
    return t0/(t1+t)

def stochastic_gradient_descent_linear_fit(n_epochs, m, X, y, X_new, t0, t1, theta_path_sgd):
    X_b = np.c_[np.ones((100,1)),X]
    # 这里增加一列是为了对应回归模型中的截距项
    X_new_b = np.c_[np.ones((2,1)),X_new]
    # 其实是 y = theta0 + theta1*x1 + theta2*x2 + ...
    # 需要用矩阵来表示 y = theta * x
    # 我们要表示theta0，所以x中加一列为1，意思是每一行的特征中我们多加了一个常量1的特征
    theta = np.random.randn(2,1)

    for epoch in range(n_epochs):
        for i in range(m):
            if epoch < 10 and i<10:
                y_predict = X_new_b.dot(theta)
                plt.plot(X_new,y_predict,'r-')
            random_index = np.random.randint(m)
            
            # 这里随机选取了一对数据样本，我们要进行一次更新
            xi = X_b[random_index:random_index+1]
            yi = y[random_index:random_index+1]

            # 计算一次样本的梯度
            gradients = 2* xi.T.dot(xi.dot(theta)-yi)

            # 更新学习率 eta
            eta = learning_schedule(t0, t1, epoch*m+i)

            # 根据梯度更新参数 theta
            theta = theta-eta*gradients
            theta_path_sgd.append(theta)

    plt.title('stochastic_gradient_descent_linear')
    return theta

def miniBatch_gradient_descent_linear_fit(n_epochs, minibatch, theta, X, y, X_new, t, t0, t1, theta_path_mgd):
    X_b = np.c_[np.ones((100,1)),X]
    # 这里增加一列是为了对应回归模型中的截距项
    X_new_b = np.c_[np.ones((2,1)),X_new]
    # 先作出原始图像
    plt.plot(X, y,'b.')
    np.random.seed(42)
    # 得到样本数据的总数
    m = len(X_b)
    for epoch in range(n_epochs):
        random_index = [np.random.randint(m) for _ in range(minibatch)]
        Xi = X[random_index]
        yi = y[random_index]
        Xi_b = np.c_[np.ones((minibatch,1)),Xi]

        # 进行一次预测，并且作图
        y_predict = X_new_b.dot(theta)
        plt.plot(X_new,y_predict,'r-')

        # 计算样本的梯度
        gradients = 2/minibatch * Xi_b.T.dot(Xi_b.dot(theta)-yi)

        # 更新学习率 eta
        eta = learning_schedule(t0, t1, epoch*minibatch+t)
        t = t + 1

        # 更新模型的参数 theta
        theta = theta - eta*gradients
        theta_path_mgd.append(theta)
        plt.xlabel('X_1')
        plt.axis([0,2,0,15])
        plt.title('miniBatch_gradient_descent_linear(eta = {})'.format(eta))
    
    return theta


    
    


