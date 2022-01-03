import numpy as np              # numpyのインポート
import matplotlib.pyplot as plt # matplotlibのpyplotをインポート

x = np.linspace(-10, 10, 100)   # -10から10までを100等分した等差数列を生成
y = 1 / (1 + np.exp(-x))        # シグモイド関数

plt.plot(x, y)

plt.ylim(-1, 2)                 # y軸の範囲
plt.xlim(-10, 10)               # x軸の範囲
plt.grid(True)                  # グリッドを表示
plt.show()
