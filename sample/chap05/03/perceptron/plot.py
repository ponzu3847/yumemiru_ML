import numpy as np
import matplotlib.pyplot as plt

# 学習データを読み込む
data = np.loadtxt('figure.csv', # 読み込むファイル
                   delimiter=',',# 区切り文字を指定
                   skiprows=1    # 1行目のタイトルを読み飛ばす
                   )
x = data[:,0:2] # 1～2列目の成分をxに代入
t = data[:,2]   # 2列目の成分をtに代入

# y軸の範囲を設定
x1 = np.arange(0, 600) 

# 分類ラベルが1のデータをドットでプロット
plt.plot(
    x[t ==  1, 0], x[t ==  1, 1], 'o'
    )
# 分類ラベルが－1のデータを▲でプロット
plt.plot(
    x[t == -1, 0], x[t == -1, 1], '^'
    )
# グラフを表示
plt.show()
