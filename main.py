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
        self.time_col = time_col
        self.state_cols = state_cols
        self.control_cols = control_cols
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
        self.degree = 3     # 多项式库阶数

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
        optimizer = Lasso(alpha=self.alpha,
                          fit_intercept=False,
                          max_iter=100000)
        poly_lib = PolynomialLibrary(degree=self.degree, include_bias=False)
        # 可选其他库：FourierLibrary, CustomLibrary
        lib = poly_lib

        self.model = SINDy(optimizer=optimizer, feature_library=lib)
        # 使用训练集训练
        dt = np.mean(np.diff(self.t))
        self.model.fit(self.X_train, t=dt, u=self.U_train, x_dot=self.X_dot_train)

    def plot_xdot(self, figsize=(14, 8), marker_size=5, file_path=None):
        """合并训练集与新数据集的导数对比图：左侧为训练集，右侧为file_path指定数据"""
        if self.model is None:
            raise ValueError("请先调用 train_model() 方法训练模型")

        # 准备训练集数据
        X_train = self.X_train
        U_train = self.U_train
        X_dot_train = self.X_dot_train
        idx_train = np.arange(X_train.shape[0])

        # 准备新数据集（file_path）
        if file_path:
            data2 = pd.read_excel(file_path, engine='openpyxl')
            t2 = data2[self.time_col].values
            X_new = data2[self.state_cols].values
            U_new = data2[self.control_cols].values
            dt2 = np.mean(np.diff(t2))
            X_dot_new = np.gradient(X_new, dt2, axis=0)
            idx_new = np.arange(X_new.shape[0])
        else:
            X_new = self.X_test
            U_new = self.U_test
            X_dot_new = self.X_dot_test
            idx_new = np.arange(X_new.shape[0])

        # 预测导数
        X_dot_train_pred = self.model.predict(X_train, u=U_train)
        X_dot_new_pred = self.model.predict(X_new, u=U_new)

        n_states = X_train.shape[1]
        plt.figure(figsize=figsize)
        for i in range(n_states):
            # 左侧：训练集
            plt.subplot(n_states, 2, 2 * i + 1)
            plt.scatter(idx_train, X_dot_train[:, i], s=marker_size, label='True Train', alpha=0.6)
            plt.scatter(idx_train, X_dot_train_pred[:, i], s=marker_size, label='Pred Train', alpha=0.6)
            if i == 0:
                plt.title('Train Set')
            plt.ylabel(f"d{self.ylabel_list[i]}/dt")
            if i == n_states - 1:
                plt.xlabel('Sample Index')
            plt.legend()

            # 右侧：新数据集
            plt.subplot(n_states, 2, 2 * i + 2)
            plt.scatter(idx_new, X_dot_new[:, i], s=marker_size, label='True New', alpha=0.6)
            plt.scatter(idx_new, X_dot_new_pred[:, i], s=marker_size, label='Pred New', alpha=0.6)
            if i == 0:
                plt.title('New Data')
            if i == n_states - 1:
                plt.xlabel('Sample Index')
            plt.legend()

        plt.tight_layout()
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
    visualizer.plot_xdot(file_path="state_and_control2.xlsx")
