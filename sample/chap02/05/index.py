import numpy as np              # numpyのインポート
import matplotlib.pyplot as plt # matplotlibのpyplotをインポート

x = np.linspace(-4, 4, 100)     # -4から4までを100等分した等差数列を生成   
y1 = 2**x                       # y1はxの2乗
y2 = 3**x                       # y2はxの3乗
y3 = 0.5**x                     # y3はxの0.5乗

plt.plot(x, y1, 'black', label='$y=2^x$')  # x、y1をプロット
plt.plot(x, y2, 'blue',  label='$y=3^x$')  # x、y2をプロット
plt.plot(x, y3, 'gray',  label='$y=0.5^x$')# x、y3をプロット
plt.ylim(-2, 6)                            # y軸の範囲
plt.xlim(-4, 4)                            # x軸の範囲
plt.grid(True)                             # グリッドを表示
plt.legend(loc='lower right')              # 凡例を右下に表示
plt.show()
