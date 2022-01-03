import numpy as np                       # numpyのインポート
import matplotlib.pyplot as plt          # matplotlib.pylabをpltとして使用する
import mpl_toolkits.mplot3d.axes3d as p3 # 3Dモジュールのインポート

def func(x1,                 # int: f(x1,x2)のx1の値
         x2                  # int: f(x1,x2)のx2の値
         ):
    '''
    3次元グラフで使用する関数

    '''
    return x1**2 + x2**2     # float: f(x1, x2)=x1^2 + x2^2

######## 実行ブロック ######################################################
if __name__ == '__main__':

    x1 = np.arange(-3, 3, 0.25)  # x1軸を生成
    x2 = np.arange(-3, 3, 0.25)  # x2軸を生成
    X1, X2 = np.meshgrid(x1, x2) # 2次元メッシュ(格子)を生成
    Y = func(X1, X2)             # 関数f(x1, x2)に配列xを代入し、-3から3までの
                                 # 0.25刻みのy値のリストを取得

    fig = plt.figure()           # 2次元のグラフを生成
    ax = p3.Axes3D(fig)          # 3次元化する
    ax.set_xlabel("x1")          # x1の軸ラベル
    ax.set_ylabel("x2")          # x2の軸ラベル
    ax.set_zlabel("f(x1, x2")    # f(x1,x2)の軸ラベル
    ax.plot_wireframe(X1,X2,Y)   # x1、x2、 f(x1,x2)の曲線をプロット
    plt.show()                   # グラフを描画
