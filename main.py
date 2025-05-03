import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pysindy import SINDy
from pysindy import PolynomialLibrary, FourierLibrary
from sklearn.linear_model import Lasso
from scipy.signal import savgol_filter


class SINDyVisualizer:
    def __init__(self, data_path, state_cols, control_cols, time_col='t'):
        """
        初始化SINDy
        :param data_path: Excel数据文件路径
        :param state_cols: 状态变量列名列表，可以多个
        :param control_cols: 控制输入列名列表
        :param time_col: 时间戳列名
        """
        # 读取数据
        self.df = pd.read_excel(data_path, engine='openpyxl')

        # 提取原始数据
        self.t = self.df[time_col].values
        self.X = self.df[state_cols].values
        self.U = self.df[control_cols].values

        # 预处理配置
        self.window_length = 5  # 滤波窗口长度
        self.polyorder = 2  # 滤波多项式阶数
        self.alpha = 0.1  # Lasso正则化系数
        self.degree = 2  # 多项式库阶数，（单纯用多项式拟合是不是有点。。。）

        # 初始化模型
        self.model = None
        self.X_dot = None

        # 用来画图的一些标签啥的
        self.ylabel_list = state_cols

    def preprocess_data(self):
        """数据预处理流程"""

        # 数据平滑处理
        # self.X = savgol_filter(self.X,
        #                        window_length=self.window_length,
        #                        polyorder=self.polyorder)

        # 简单计算数值导数
        dt = np.mean(np.diff(self.t))  # 其实是一定的离散间隔
        self.X_dot = np.gradient(self.X, dt, axis=0)

    def train_model(self):
        """训练SINDy模型"""
        # 配置模型
        optimizer = Lasso(alpha=self.alpha, max_iter=100000)
        poly_lib = PolynomialLibrary(degree=self.degree)  # 多项式
        trig_lib = FourierLibrary(n_frequencies=3)
        lib = poly_lib + trig_lib

        # 训练模型
        self.model = SINDy(optimizer=optimizer,
                           feature_library=lib)
        self.model.fit(self.X, t=np.mean(np.diff(self.t)), u=self.U, x_dot=self.X_dot)

    def plot_combined_comparison(self, figsize=(10, 6)):
        # Predict derivatives for all data
        X_dot_pred = self.model.predict(self.X, u=self.U)
        dt = self.t[1] - self.t[0]

        # One-step-ahead prediction using true state
        n = len(self.t)
        X_pred = np.zeros_like(self.X)
        X_pred[0] = self.X[0]
        for i in range(n - 1):
            X_pred[i+1] = self.X[i] + X_dot_pred[i] * dt

        # Plotting
        plt.figure(figsize=figsize)
        n_states = self.X.shape[1]
        for i in range(n_states):
            # Derivative comparison (left)
            plt.subplot(n_states, 2, 2*i + 1)
            plt.plot(self.t, self.X_dot[:, i], 'r--', label='True derivative')
            plt.plot(self.t, X_dot_pred[:, i], 'b-', label='Predicted derivative')
            plt.ylabel(f"d{self.ylabel_list[i]}/dt")
            plt.legend()

            # One-step state prediction (right)
            plt.subplot(n_states, 2, 2*i + 2)
            plt.plot(self.t, self.X[:, i], 'r--', label='True state')
            plt.plot(self.t, X_pred[:, i], 'b-', label='1-step prediction')
            plt.ylabel(self.ylabel_list[i])
            plt.legend()

        plt.tight_layout()
        plt.suptitle('Derivative and One-Step Prediction Comparison', y=1.02)
        plt.xlabel('Time')
        plt.show()

    def plot_residuals(self, figsize=(8, 6)):
        """绘制预测残差分布"""
        if self.model is None:
            raise ValueError("请先调用train_model()方法训练模型")

        residuals = self.X_dot - self.model.predict(self.X, u=self.U)

        plt.figure(figsize=figsize)
        plt.hist(residuals.flatten(), bins=50, density=True, alpha=0.6)
        plt.title('Prediction Residual Distribution')
        plt.xlabel('Residual Value')
        plt.ylabel('Probability Density')

    def show_model(self):
        """打印模型方程"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        self.model.print()


# 使用示例 =====================================================
if __name__ == "__main__":
    # 初始化可视化工具
    visualizer = SINDyVisualizer(
        data_path="state_and_control.xlsx",
        state_cols=['beta', 'omega', 'v'],
        control_cols=['F_xf', 'F_xr', 'delta_f', 'delta_r'],
        time_col='time'
    )

    # 数据预处理
    visualizer.preprocess_data()

    # 训练模型
    visualizer.train_model()

    # 显示模型方程
    visualizer.show_model()

    # 绘制合并对比图和残差图
    visualizer.plot_combined_comparison()
    # visualizer.plot_residuals()
    plt.show()