# 这是一个使用SINDy拟合非线性动力学的项目

**在“state_and_control.xlsx”文件中，包含了时间序列“time”、控制序列和状态序列**

## 状态序列$\text{X(t)}$

$$
\text{X}(t) = [\beta(t), \omega(t), V(t)]
$$

分别表示车辆的**质心侧偏角**（rad），**横摆角速度**（rad/s），**合速度**（m/s）

## 控制序列

$$
\text{U}(t) = [F_{xf}(t), F_{xr}(t), \delta_f(t), \delta_r(t)]
$$



分别表示车辆的**前轴纵向力**，**后轴纵向力**，**前轮转向角**，**后轮转向角**（rad）

*实际输入到车辆时需转换为单个车轮的控制量，举例：*
$$
T_{xfl} = \frac{F_{xf}}{2} \cdot r
$$

$$
\delta_{fl} = \delta_f \cdot \frac{180}{\pi}
$$

*以及需要满足最大控制量约束*：
$$
\delta_{fl} \leq 35 \textdegree
$$


