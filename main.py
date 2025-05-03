import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.linear_model import Lasso
from pysindy import SINDy, PolynomialLibrary
from pysindy import CustomLibrary

file_path = "state_and_control.xlsx"
df = pd.read_excel(file_path, sheet_name='Sheet1', engine='openpyxl')

# 提取数据
t = df['time'].values  # 时间序列
X = df[['beta', 'omega', 'v']].values  # 状态
U = df[['F_xf', 'F_xr', 'delta_f', 'delta_r']].values  # 控制量

# 状态导数X_dot
dt = np.mean(np.diff(t))  # 这里实际上是固定的
# print(dt)
X_dot = np.gradient(X, dt, axis=0)

# 构建候选函数库
custom_functions = [
    
]
model = SINDy(
    optimizer=Lasso(alpha=0.1),
    feature_library=PolynomialLibrary(degree=2)
)
model.fit(X, t=dt, u=U)
model.print()

