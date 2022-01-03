import numpy as np
import matplotlib.pyplot as plt

# データの分布を表す関数
def distribution(x):
     return -0.09 * (x + x**2)

# xの値を等間隔に20個生成する
x = np.linspace(start=-3,    # 数列の始点
                stop=3,      # 数列の終点
                num=20       # 等差数列の要素数
                )
# 関数が出力値したy値に若干の変動を加える
y = distribution(x) + np.random.randn(x.size)*0.1

# 学習データをプロット
plt.plot(x, y, '^')
# x軸の値として-3から3までの等差数列を生成
x0 = np.linspace(start=-3, stop=3, num=100)

plt.grid(True)
plt.show()
