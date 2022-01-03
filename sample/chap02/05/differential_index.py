import numpy as np              # numpyのインポート
import matplotlib.pyplot as plt # matplotlibのpyplotをインポート

x = np.linspace(-5, 5, 100)     # -4から4までを100等分した等差数列を生成 
a = 2                           # aの値をセット
y = a**x                        # yはaのx乗
dy = np.log(a) * y              # 微分する

plt.plot(x, y)                  # x、yをプロット
plt.plot(x, dy,  linestyle='dashed')# 微分した結果をプロット
plt.ylim(-1, 8)                 # y軸の範囲
plt.xlim(-5, 5)                 # x軸の範囲
plt.grid(True)                  # グリッドを表示
plt.show()
