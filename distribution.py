import pandas as pd
import matplotlib.pyplot as plt


def compare_variable_distributions(file1, file2, state_cols, control_cols, time_col='time'):

    # 读取数据
    df1 = pd.read_excel(file1, engine='openpyxl')
    df2 = pd.read_excel(file2, engine='openpyxl')

    # 提取状态量和控制量
    X1 = df1[state_cols].values
    U1 = df1[control_cols].values
    X2 = df2[state_cols].values
    U2 = df2[control_cols].values

    # 创建子图
    fig, axes = plt.subplots(2, max(len(state_cols), len(control_cols)), figsize=(14, 8), sharex=False)
    
    # 绘制状态量分布
    for i, label in enumerate(state_cols):
        ax = axes[0, i]  # 第一行用于状态量
        ax.hist(
            X1[:, i],
            bins=30,
            alpha=0.5,
            label=f"File1 State: {label}",
            density=True,
            histtype='stepfilled',
            edgecolor='black'
        )
        ax.hist(
            X2[:, i],
            bins=30,
            alpha=0.5,
            label=f"File2 State: {label}",
            density=True,
            histtype='stepfilled',
            edgecolor='black'
        )
        ax.set_title(f"Distribution of State: {label}")
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        # ax.legend()

    # 绘制控制量分布
    for i, label in enumerate(control_cols):
        ax = axes[1, i]  # 第二行用于控制量
        ax.hist(
            U1[:, i],
            bins=30,
            alpha=0.5,
            label=f"File1 Control: {label}",
            density=True,
            histtype='stepfilled',
            edgecolor='black'
        )
        ax.hist(
            U2[:, i],
            bins=30,
            alpha=0.5,
            label=f"File2 Control: {label}",
            density=True,
            histtype='stepfilled',
            edgecolor='black'
        )
        ax.set_title(f"Distribution of Control: {label}")
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        # ax.legend()
    
    # 隐藏多余的子图
    for i in range(len(state_cols), len(axes[0])):
        axes[0, i].axis('off')
    for i in range(len(control_cols), len(axes[1])):
        axes[1, i].axis('off')
    
    # 调整布局并显示
    plt.tight_layout()
    
    # 调整布局并显示
    plt.tight_layout()

    # 调整布局并显示
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    file1 = "state_and_control.xlsx"
    file2 = "state_and_control2.xlsx"

    compare_variable_distributions(file1, file2,
                                   ['beta', 'omega', 'v'],
                                   control_cols=['F_xf', 'F_xr', 'delta_f', 'delta_r'])