import numpy as np              # numpyのインポート
import matplotlib.pyplot as plt # matplotlibのpyplotをインポート

x1 = np.linspace(-8, 8, 100)    # -8から8までを100等分した等差数列を生成
y1 = 2**x1                      # y1は2のx乗

x2 = np.linspace(0.001, 8, 100) # np.log(0)はエラーになるのでx2に0を含めない
y2 = np.log(x2) / np.log(2)     # 底が2、指数がx2のときの対数関数を計算

plt.plot(x1, y1, 'black', label='$y=2^x$')               # 指数関数をプロット
plt.plot(x2, y2, 'blue',  label='$y=log(2)x$')           # 対数関数をプロット
plt.plot(x1, x1, 'black', label='$y=x$', linestyle='dashed') # y=xの直線
plt.ylim(-8, 8)                                          # y軸の範囲
plt.xlim(-8, 8)                                          # x軸の範囲
plt.grid(True)                                           # グリッドを表示
plt.legend(loc='lower right')                            # 凡例を右下に表示
plt.show()
