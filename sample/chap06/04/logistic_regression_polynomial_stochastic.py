import numpy as np
import matplotlib.pyplot as plt

def standardize(x):
    '''
        標準化を行う
        
        パラメーター
        ------------
        x : 標準化前のx1、x2
    '''
    x_mean = x.mean(axis=0)       # 列ごとの平均値を求める
    std = x.std(axis=0)           # 列ごとの標準偏差を求める
    
    return (x - x_mean) / std     # 標準化した値を返す


def create_matrix(x):
    '''
        バイアス項x0、x1、x2、x1を2乗した項の行列を作成        
    '''
    x0 = np.ones([x.shape[0], 1])  # バイアス項を作成する
    x3 = x[:,0,np.newaxis] ** 2    # x1を2乗した項を作成する
    return np.hstack([x0, x, x3])

def sigmoid(x, parameter):
    '''
        シグモイド関数        
    '''
    return 1 / (1 + np.exp(-np.dot(x, parameter)))

def logistic_polynomial_stochastic(matrix_x, t):
    '''
        多項式回帰で最適化を行う
        
        パラメーター
        ------------
        matrix_x : x0、x1、x2、x1を2乗した項のの行列
        t        : 分類値t
    '''
    # Φ0、Φ1、Φ2、Φ3を初期化
    parameter = np.random.rand(4)
    # 学習率を0.001に設定
    LNR = 1e-3
    # 更新回数
    loop = 3000
    # 更新回数をカウントする変数を0で初期化
    count = 0

    # 学習をloop回繰り返す
    for i in range(loop):
        # 確率的勾配降下法でパラメーターを更新
        p = np.random.permutation(matrix_x.shape[0])
        for tmp_x, tmp_t in zip(matrix_x[p,:], t[p]):
            parameter = parameter - LNR * (sigmoid(tmp_x,
                                                   parameter) - tmp_t
                                           ) * tmp_x

        # カウンター変数の値を1増やす
        count += 1
        # 処理結果を出力
        print(
            '({}) parameter: {}'.format(count, parameter)
            )
    
    # Φ0、Φ1、Φ2、Φ3の行ベクトルを戻り値として返す
    return parameter

######## 実行ブロック ######################################################
if __name__ == '__main__':
    
    # 学習データを読み込む
    data = np.loadtxt('classification2.csv', # 読み込むファイル
                      dtype='int',           # データ型を指定
                      delimiter=',',         # 区切り文字を指定
                      skiprows=1             # 1行目のタイトルを読み飛ばす
                      )
    x = data[:,0:2]                          # x1、x2を行列xに代入
    t = data[:,2]                            # 3列目の成分をtに代入
    standardized_x = standardize(x)          # xのすべての成分を標準化
    matrix_x = create_matrix(standardized_x) # 標準化したxにバイアス項を追加する
    # パラメーターの値を求める
    parameter = logistic_polynomial_stochastic(matrix_x, t)

    # x軸の値として-2から2までの等差数列を生成
    x0 = np.linspace(start=-2,              # 数列の始点
                     stop=2,                # 数列の終点
                     num=100                # 等差数列の要素数
                     )
                     
    # t == 1のデータをプロット
    plt.plot(standardized_x[t == 1, 0],
             standardized_x[t == 1, 1], 'o')
    # t == 0のデータをプロット
    plt.plot(standardized_x[t == 0, 0],
             standardized_x[t == 0, 1], '^')
    # 決定境界をプロット
    plt.plot(x0,
             -(parameter[0] + parameter[1] * x0
             + parameter[3] * x0 ** 2) / parameter[2],
             linestyle='solid'
             )
    plt.grid(True)
    plt.show()
