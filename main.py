import subprocess  
  
# 任务一，实现三种梯度下降法的路径对比  
subprocess.run(["python", "linear_regression_main.py"], check=True)

# 任务二，调整多项式拟合的degree，观察过拟合、欠拟合的结果
# 任务三，观察过拟合时的标志
subprocess.run(["python", "polynomial_regression_main.py"], check=True)
