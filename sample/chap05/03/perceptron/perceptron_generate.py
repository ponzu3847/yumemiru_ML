import numpy as np
import matplotlib.pyplot as plt
import random

def separatrix(x, y):
    '''
        データx1、x2の座標を分割する直線の式
        
    '''
    return 3*x - 5*y - 1  #  境界線を3x-5y-1とする

def convert(x1, x2):
    '''
        データx1、x2をバイアス項を加えたベクトルにする
        
    '''
    x0 = np.ones(x1.shape)        # バイアス項を作成する
    return np.array([x0, x1, x2]) # バイアス項、x1、x2の行ベクトルを返す

def classify(x, w):
    '''
        分類関数
        
        パラメーター
        ------------
        x : x1、x2の行列
        w : w1、w2のベクトル
    '''
    if np.dot(w, x) >= 0:
        return 1  # w・x≧0 なら1を返す
    else:
        return -1 # w・x＜0 なら－1を返す


def learn_3weights(X, T, N):
    '''
        更新式で重みを学習する
        
        パラメーター
        ------------
        X :  x1、x2の行列
        T : ラベル
        N : データの個数
        
    '''
    w = np.zeros(3)        # 重みｗ1、w2、w3を0で初期化
    while True:
      lst = list(range(N)) # 0からN-1までの配列を作る
      random.shuffle(lst)  # バラバラに並べ替える
      misses = 0           # 予測を外した回数をカウントする変数

      for n in lst:        # 並べ替えたリストから順番に取り出す
        x_1, x_2 = X[n, :] # 行列Xのn行から成分x1、x2を取り出す
        t_n = T[n]         # ベクトルTのn行の成分を取り出す

        # 分類関数で分類する
        predict = classify(convert(x_1, x_2), # ベクトルを作る
                           w                  # 重みのベクトル
                           )

        # 分類関数の結果が不正解なら重みを更新する
        if predict != t_n:
          w += t_n * convert(x_1, x_2)
          misses += 1 # 更新後missesを1増やす

      # 不正解が無くなったら学習終了
      if misses == 0:
        break
    
    return w # 重みベクトルを返す

######## 実行ブロック ######################################################
if __name__ == '__main__':

    # データの個数
    num = 300
    # データの乱数列を固定する
    np.random.seed(0)
    # ランダムな N×2の行列を生成
    X = np.random.randn(num, 2)
    # 区分線よりも上にある点にラベルとして+1、
    # 下にある場合は－1を振る
    T = np.array([1 if separatrix(x1, x2) > 0 else -1
                  for x1, x2 in X
                  ])

    # 重みw1、w2、w3の値を学習する
    w = learn_3weights(X,  # x1、x2の行列
                       T,  # ラベル
                       num # データの個数
                       )    
    
    # -4から4まで0.03刻みの等差数列を生成
    prog = np.arange(-4, 4, 0.03)
    # 等差数列をx、yの値にして格子座標を生成
    xlist, ylist = np.meshgrid(prog, prog)
    # xlist、ylistのデータ点に更新後の重みを適用し、配列で取得
    predict = [np.sign(  # sign()で正の値は1、負の値は-1として取得
                np.dot(w,              # 更新後の重み
                       convert(x1, x2) # バイアス項、x1、x2の行列を生成
                       )
             )
             for x1, x2 in zip(xlist, ylist)]

    # 学習結果で2次元の平面を色分けする
    plt.pcolormesh(xlist,    # x軸
                   ylist,    # y軸
                   predict,  # 重み適用後の値(-1,または1)をカラー値にする
                   alpha=0.1,# 透明度(0～1)
                   )

    # 分類ラベルが1のデータをプロット
    plt.plot(X[T== 1, 0],    # x軸
             X[T== 1, 1],    # y軸
             's',            # 四角形
             color='black',  # ブラック
             markersize=2    # サイズは2
             )

    # 分類ラベルが－1のデータをプロット
    plt.plot(X[T==-1, 0],    # x軸
             X[T==-1, 1],    # y軸
             'o',            # ドット
             color='red',    # 赤
             markersize=2    # サイズは2
             )

    plt.grid(True)
    plt.show()
