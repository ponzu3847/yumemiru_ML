import numpy as np
import matplotlib.pyplot as plt

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

def learn_weights(x, t):
    '''
        更新式で重みを学習する
        
        パラメーター
        ------------
        x : x1、x2の行列
        w : w1、w2のベクトル
    '''
    w = np.random.rand(2) # 重みの初期化  
    loop = 5              # 繰り返し回数
    count = 0             # 繰り返しの回数をカウントする変数

    # 指定した回数だけ重みの学習を繰り返す
    for i in range(loop):
        # ベクトルx、tから成分を取り出す
        for element_x, element_t in zip(x, t):
            # 分類関数の出力が異なる場合は重みを更新する
            if classify(element_x, w) != element_t:
                w = w + element_t * element_x
                print('更新後のw = ', w)
        count += 1
        # ログの出力
        print('[{}周目]: w = {}***'.format(count, w))

    return w

######## 実行ブロック ######################################################
if __name__ == '__main__':

    # 学習データを読み込む
    data = np.loadtxt('figure.csv',  # 読み込むファイル
                       delimiter=',',# 区切り文字を指定
                       skiprows=1    # 1行目のタイトルを読み飛ばす
                       )
    x = data[:,0:2]                  # 1～2列目の成分をxに代入
    t = data[:,2]                    # 2列目の成分をtに代入
    # 重みw1、w2の値を求める
    w = learn_weights(x, t)
    
    # 軸の範囲を設定
    x1 = np.arange(0, 600)    
    # 分類ラベルが1のデータをドットでプロット
    plt.plot(
        x[t ==  1, 0], x[t ==  1, 1], 'o'
        )
    # 分類ラベルが－1のデータを▲でプロット
    plt.plot(
        x[t == -1, 0], x[t == -1, 1], '^'
        )
    # 境界線をプロット
    plt.plot(
        x1, -w[0] / w[1] * x1, linestyle='solid'
        )
    plt.grid(True)
    plt.show()
    
    classify([200, 100],w)
