import numpy as np
import matplotlib.pyplot as plt

def standardize(x):
    '''
        標準化を行う
        
        パラメーター
        ------------
        x : 標準化前のx1、x2
    '''
    x_mean = x.mean(axis=0)               # 列ごとの平均値を求める
    std = x.std(axis=0)                   # 列ごとの標準偏差を求める
    
    return (x - x_mean) / std             # 標準化した値を返す

######## 実行ブロック ######################################################
if __name__ == '__main__':
    
    # 学習データを読み込む
    data = np.loadtxt('classification.csv',      # 読み込むファイル
                      dtype='int',               # データ型を指定
                      delimiter=',',             # 区切り文字を指定
                      skiprows=1                 # 1行目のタイトルを読み飛ばす
                      )
    x = data[:,0:2]                              # x1、x2を行列xに代入
    t = data[:,2]                                # 3列目の成分をtに代入
    standardized_x = standardize(x)              # xのすべての成分を標準化

    # t == 1のデータをプロット
    plt.plot(standardized_x[t == 1, 0],
             standardized_x[t == 1, 1], 'o')
    # t == 0のデータをプロット
    plt.plot(standardized_x[t == 0, 0],
             standardized_x[t == 0, 1], '^')
    plt.grid(True)
    plt.show()
