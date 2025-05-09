import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
from pysindy import PolynomialLibrary, FourierLibrary, CustomLibrary
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNet

# 自定义的函数库
def create_custom_functions():

    custom_functions = [
        lambda x: np.arctan(x),
        lambda x: x * np.arctan(x),
        lambda x: x * np.sin(x),
        lambda x: x * np.cos(x),
    ]
    return custom_functions

class SINDyVisualizer:
    def __init__(self, data_path, state_cols, control_cols, time_col='time', random_state=None):
        """
        初始化SINDy可视化工具，使用全量数据训练
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

        # 计算全量真实导数
        dt = np.mean(np.diff(self.t))
        self.X_dot = np.gradient(self.X, dt, axis=0)
        self.ylabel_list = state_cols

        # 归一化器
        self.u_scaler = MinMaxScaler()
        self.x_scaler = MinMaxScaler()
        self.U = self.u_scaler.fit_transform(self.U)
        self.X = self.x_scaler.fit_transform(self.X)

        # 模型与参数
        self.degree = 3
        self.n_frequency = 2
        self.model = None

    def train_model(self):
        """使用全量数据训练SINDy模型"""
        # optimizer = Lasso(alpha=0.005, fit_intercept=False, max_iter=1000000)
        # optimizer = ps.STLSQ(threshold=0.15)
        optimizer = ElasticNet(alpha=0.005, l1_ratio=0.9, fit_intercept=False)
        poly_lib = PolynomialLibrary(degree=self.degree, include_bias=False)
        trig_lib = FourierLibrary(n_frequencies=self.n_frequency)
        #
        custom_functions = create_custom_functions()
        cus_lib = CustomLibrary(library_functions=custom_functions)
        lib = poly_lib
        lib = poly_lib + trig_lib + cus_lib

        self.model = ps.SINDy(optimizer=optimizer, feature_library=lib)
        dt = np.mean(np.diff(self.t))
        self.model.fit(self.X, t=dt, u=self.U, x_dot=self.X_dot)

    def plot_xdot(self, figsize=(14, 8), marker_size=5, file_path=None):
        """左：全量训练集；右：新数据集"""
        if self.model is None:
            raise ValueError("请先调用 train_model() 方法训练模型")

        # 左：全量训练集
        X_train = self.X
        U_train = self.U
        X_dot_true = self.X_dot
        idx_train = np.arange(X_train.shape[0])
        X_dot_train_pred = self.model.predict(X_train, u=U_train)

        # 右：新数据或同X
        if file_path:
            data2 = pd.read_excel(file_path, engine='openpyxl')
            t2 = data2[self.time_col].values
            X_new = data2[self.state_cols].values
            U_new = data2[self.control_cols].values
            dt2 = np.mean(np.diff(t2))
            X_dot_new_true = np.gradient(X_new, dt2, axis=0)
            # 也归一化
            U_new = self.u_scaler.transform(U_new)
            X_new = self.x_scaler.transform(X_new)
        else:
            X_new = self.X
            U_new = self.U
            X_dot_new_true = self.X_dot
        idx_new = np.arange(X_new.shape[0])
        X_dot_new_pred = self.model.predict(X_new, u=U_new)

        # 绘图
        n_states = X_train.shape[1]
        plt.figure(figsize=figsize)
        for i in range(n_states):
            # 左
            plt.subplot(n_states, 2, 2*i+1)
            plt.scatter(idx_train, X_dot_true[:, i], s=marker_size, label='True Train', alpha=0.6)
            plt.scatter(idx_train, X_dot_train_pred[:, i], s=marker_size, label='Pred Train', alpha=0.6)
            if i==0: plt.title('Train Set')
            plt.ylabel(f"d{self.ylabel_list[i]}/dt")
            if i==n_states-1: plt.xlabel('Sample Index')
            plt.legend()
            # 右
            plt.subplot(n_states, 2, 2*i+2)
            plt.scatter(idx_new, X_dot_new_true[:, i], s=marker_size, label='True New', alpha=0.6)
            plt.scatter(idx_new, X_dot_new_pred[:, i], s=marker_size, label='Pred New', alpha=0.6)
            if i==0: plt.title('New Data')
            if i==n_states-1: plt.xlabel('Sample Index')
            plt.legend()

        plt.tight_layout()
        # plt.savefig("aa.png")
        plt.show()

    def show_model(self):
        """打印并显示已识别的SINDy模型方程"""
        if self.model is None:
            raise ValueError("请先调用 train_model() 方法训练模型")
        self.model.print()


if __name__ == "__main__":

    train_file = "state_and_control.xlsx"
    test_file = "state_and_control2.xlsx"


    visualizer = SINDyVisualizer(
        data_path=train_file,
        state_cols=['beta', 'omega', 'v'],
        control_cols=['F_xf', 'F_xr', 'delta_f', 'delta_r'],
        time_col='time',
        random_state=None  # 每次运行随机生成
    )

    visualizer.train_model()
    visualizer.show_model()
    # 可通过 marker_size 参数调整点大小
    visualizer.plot_xdot(file_path=test_file)
