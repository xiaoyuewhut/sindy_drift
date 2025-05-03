Github不能正常显示LaTeX格式公式，推荐使用Typora阅读。https://pan.xunlei.com/s/VONSf5lUM8w5fKDwGby5F8nzA1?pwd=bghe

[toc]

# 这是一个使用SINDy识别漂移工况非线性动力学的项目

**在“state_and_control.xlsx”文件中，包含了时间序列“time”、控制序列和状态序列**。

“state_and_control2.xlsx”这个文件中，存放在另一组仿真数据，其状态量和前者相似，但控制量有很大不同，用前者训练的模型应当在后者也有较好的效果。

## 需要依赖项

python == 3.10

numpy == 1.24.4 （必须是这个版本）

pysindy

openyxl

pandas

## 状态序列$X(t)$

$$
X(t) = [\beta(t), \omega(t), V(t)]
$$

分别表示车辆的**质心侧偏角**（rad），**横摆角速度**（rad/s），**合速度**（m/s）

## 控制序列

$$
\text{U}(t) = [F_{xf}(t), F_{xr}(t), \delta_f(t), \delta_r(t)]
$$

分别表示车辆的**前轴纵向力**，**后轴纵向力**，**前轮转向角**，**后轮转向角**（rad）

实际输入到车辆时需转换为单个车轮的控制量，举例：
$$
T_{xfl} = \frac{F_{xf}}{2} \cdot r
$$

$$
\delta_{fl} = \delta_f \cdot \frac{180}{\pi}
$$

以及需要满足最大控制量约束：
$$
\delta_{fl} \leq 35 \textdegree.
$$

## 应不应该设置常数项？

——可能**不应该**，这个在以下代码中设置：

```
optimizer = Lasso(..., fit_intercept=False)
poly_lib = PolynomialLibrary(..., include_bias=False)
```

如果设置了常数项，即使状态量和控制量全为0，状态变化率也不为0，和实际不符。

## 需要注意的

* 对于实时控制，可能更应该关注的是短期预测效果$\dot{X}(t)$，而不是通过数值积分预测的长期状态量拟合效果$X(t)$。
* 数据由CarSim直接导出，理论上不存在**噪声**，需不需要提前滤波平滑很难说。
* 离散步长严格**0.05s**。
