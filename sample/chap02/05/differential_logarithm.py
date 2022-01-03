import numpy as np              # numpyのインポート
import matplotlib.pyplot as plt # matplotlibのpyplotをインポート

x = np.linspace(0.0001, 5, 100) # 0.0001から4までを100等分した等差数列を生成
y = np.log(x)                   # eを底とするxの対数関数  
dy = 1 / x                      # 対数関数の微分

plt.plot(x, y)                  # f'(x) = log x
plt.plot(x, dy, linestyle='--') # f'(x) = 1/x
plt.ylim(-8, 8)                 # y軸の範囲
plt.xlim(-1, 5)                 # x軸の範囲
plt.grid(True)                  # グリッドを表示
plt.show()
