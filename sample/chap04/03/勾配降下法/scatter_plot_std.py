import numpy as np                    # NumPyをインポート
import matplotlib.pyplot as plt       # Matplotlibをインポート

def standardize(x):                   # xは標準化前のデータ
    x_mean = x.mean()                 # 平均値を求める
    std = x.std()                     # 標準偏差を求める
    
    return (x - x_mean) / std         # 標準化した値を返す

data = np.loadtxt(fname='access.csv', # 読み込むファイル
                  dtype='int',        # データ型を指定
                  delimiter=',',      # 区切り文字を指定
                  skiprows=1          # 1行目のタイトルを読み飛ばす
                  )
x = data[:,0]                         # 1列目の成分をxに代入
y = data[:,1]                         # 2列目の成分をyに代入
standardized_x = standardize(x)

plt.plot(standardized_x,              # x軸に標準化したxを割り当てる
         y,                           # y軸に割り当てるデータ
         'o'                          # ドット(点)をプロットする
         )
plt.grid(True)
plt.show()                            # グラフを表示
