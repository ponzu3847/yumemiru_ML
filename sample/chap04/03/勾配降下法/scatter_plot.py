import numpy as np                    # NumPyをインポート
import matplotlib.pyplot as plt       # Matplotlibをインポート

data = np.loadtxt(fname='access.csv', # 読み込むファイル
                  dtype='int',        # データ型を指定
                  delimiter=',',      # 区切り文字を指定
                  skiprows=1          # 1行目のタイトルを読み飛ばす
                  )
x = data[:,0]                         # 1列目の成分をxに代入
y = data[:,1]                         # 2列目の成分をyに代入

plt.plot(x,                           # x軸に割り当てるデータ
         y,                           # y軸に割り当てるデータ
         'o'                          # ドット(点)をプロットする
         ) 
plt.grid(True)
plt.show()                            # グラフを表示
