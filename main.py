import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pysindy import SINDy
from pysindy import PolynomialLibrary, FourierLibrary, CustomLibrary
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split


class SINDyVisualizer:
    def __init__(self, data_path, state_cols, control_cols, time_col='time', train_ratio=0.7, random_state=None):
        """
        初始化SINDy可视化工具
        :param data_path: Excel数据文件路径
        :param state_cols: 状态变量列名列表
        :param control_cols: 控制输入列名列表
        :param time_col: 时间戳列名
        :param train_ratio: 训练集比例 (0-1)
        :param random_state: 随机种子，若为 None 则每次自动生成
        """
        # 读取数据
        self.df = pd.read_excel(data_path, engine='openpyxl')

        # 提取原始数据
        self.t = self.df[time_col].values
        self.X = self.df[state_cols].values
        self.U = self.df[control_cols].values

        # 划分训练和测试集比例
        self.train_ratio = train_ratio
        # 若未提供 random_state，则随机生成一个
        if random_state is None:
            self.random_state = np.random.randint(0, 2**31 - 1)
        else:
            self.random_state = random_state

        # 预处理配置
        self.alpha = 0.001  # Lasso正则化系数
        self.degree = 2     # 多项式库阶数

        # 初始化模型和数据分割容器
        self.model = None
        self.X_dot = None
        self._split_data()

        # 标签
        self.ylabel_list = state_cols

    def _split_data(self):
        """计算数值导数并进行训练/测试集划分"""
        # 计算数值导数
        dt = np.mean(np.diff(self.t))
        X_dot = np.gradient(self.X, dt, axis=0)

        # 按时序随机抽样分割
        indices = np.arange(len(self.t))
        train_idx, test_idx = train_test_split(
            indices,
            train_size=self.train_ratio,
            random_state=self.random_state,
            shuffle=True
        )

        # 存储分割后数据
        self.idx_train = np.sort(train_idx)
        self.idx_test = np.sort(test_idx)
        self.X_train = self.X[self.idx_train]
        self.U_train = self.U[self.idx_train]
        self.X_dot_train = X_dot[self.idx_train]
        self.X_test = self.X[self.idx_test]
        self.U_test = self.U[self.idx_test]
        self.X_dot_test = X_dot[self.idx_test]

    def train_model(self):
        """训练SINDy模型"""
        optimizer = Lasso(alpha=self.alpha, max_iter=100000)
        poly_lib = PolynomialLibrary(degree=self.degree)
        # 可选其他库：FourierLibrary, CustomLibrary
        lib = poly_lib

        self.model = SINDy(optimizer=optimizer, feature_library=lib)
        # 使用训练集训练
        dt = np.mean(np.diff(self.t))
        self.model.fit(self.X_train, t=dt, u=self.U_train, x_dot=self.X_dot_train)

    def plot_xdot_comparison(self, figsize=(14, 6), marker_size=5):
        """绘制训练集和测试集的x_dot真实值与预测值对比"""
        if self.model is None:
            raise ValueError("请先调用 train_model() 方法训练模型")

        # 预测训练和测试导数
        xdot_train_pred = self.model.predict(self.X_train, u=self.U_train)
        xdot_test_pred = self.model.predict(self.X_test, u=self.U_test)

        n_states = self.X.shape[1]
        plt.figure(figsize=figsize)
        for i in range(n_states):
            # 左侧：训练集
            plt.subplot(n_states, 2, 2*i + 1)
            plt.scatter(self.idx_train, self.X_dot_train[:, i], s=marker_size, label='True train', alpha=0.6)
            plt.scatter(self.idx_train, xdot_train_pred[:, i], s=marker_size, label='Pred train', alpha=0.6)
            plt.ylabel(f"d{self.ylabel_list[i]}/dt")
            if i == 0:
                plt.title('Training Set')
            if i == n_states - 1:
                plt.xlabel('Sample Index')
            plt.legend()

            # 右侧：测试集
            plt.subplot(n_states, 2, 2*i + 2)
            plt.scatter(self.idx_test, self.X_dot_test[:, i], s=marker_size, label='True test', alpha=0.6)
            plt.scatter(self.idx_test, xdot_test_pred[:, i], s=marker_size, label='Pred test', alpha=0.6)
            if i == 0:
                plt.title('Test Set')
            if i == n_states - 1:
                plt.xlabel('Sample Index')
            plt.legend()

        plt.tight_layout()
        plt.suptitle('Derivative Comparison: Train vs Test', y=1.02)
        plt.show()

    def show_model(self):
        """打印模型方程"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        self.model.print()


# 使用示例 =====================================================
if __name__ == "__main__":
    visualizer = SINDyVisualizer(
        data_path="state_and_control.xlsx",
        state_cols=['beta', 'omega', 'v'],
        control_cols=['F_xf', 'F_xr', 'delta_f', 'delta_r'],
        time_col='time',
        train_ratio=0.6,
        random_state=None  # 每次运行随机生成
    )

    visualizer.train_model()
    visualizer.show_model()
    # 可通过 marker_size 参数调整点大小
    visualizer.plot_xdot_comparison(marker_size=5)
