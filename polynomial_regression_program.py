import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

colors = ['g', 'r-+', 'b--']

def polynomial_regression_fit(X, y, X_new, degree):
    # 生成二次多项式特征，不包含常数项
    poly_features = PolynomialFeatures(degree, include_bias = False)

    # 这里的 X_poly 包含了一次项和二次项的特征，[X] -> [X, X^2]
    # 通过将X转换为X_poly（包含X和X^2），我们实际上是将一个非线性回归问题转化为了一个线性回归问题，从而可以利用线性回归模型的优点来解决它
    X_poly = poly_features.fit_transform(X)

    # 创建实例进行拟合
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly,y)

    # 输出当前的拟合信息
    print(f'当前度数为：{degree}')
    print('coef_ = ')
    print (lin_reg.coef_)
    print('intercept_ = ')
    print (lin_reg.intercept_)

    # 进行预测
    X_new_poly = poly_features.transform(X_new)
    y_new = lin_reg.predict(X_new_poly)

    # 进行可视化操作
    color_index = degree%4
    chart(X_new, y_new, colors[color_index], str(f'degree = {degree}'))

def chart(X, y, color, label):
    plt.plot(X, y, color, label = label)
    plt.axis([-3,3,-5,10])
    plt.legend()

def plot_learning_curves(model,X,y):
    # 数据分割
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size =0.2,random_state=100)

    # 空列表用于存储训练误差和验证误差
    train_errors,val_errors = [],[]
    for m in range(1,len(X_train)):
        model.fit(X_train[:m],y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m],y_train_predict[:m]))
        val_errors.append(mean_squared_error(y_val,y_val_predict))
        
    plt.plot(np.sqrt(train_errors),'r-+',linewidth = 2,label = 'train_error')
    plt.plot(np.sqrt(val_errors),'b-',linewidth = 3,label = 'val_error')
    plt.xlabel('Trainsing set size')
    plt.ylabel('RMSE')
    plt.legend()


