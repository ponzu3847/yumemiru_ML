import numpy as np
import matplotlib.pyplot as plt

def standardize(x):
    '''
        標準化を行う
        
        パラメーター
        ------------
        x : 標準化前のx
    '''
    x_mean = x.mean()                     # 平均値を求める
    std = x.std()                         # 標準偏差を求める
    
    return (x - x_mean) / std             # 標準化した値を返す

def create_matrix(standardized_x):
    '''
        全ての要素が1の行ベクトル、
        standardized_x、
        standardized_xの二乗
        を行方向に連結して行列を作成し、
        これを転置して返す
        
        パラメーター
        ------------
        standardized_x : 標準化後のxの行ベクトル
    '''
    return np.vstack(
        [np.ones(standardized_x.size), # 要素の値がすべて1の行ベクトル
                                       # (要素数はstandardized_xと同じ)
         standardized_x,               # 標準化したxの行ベクトル
         standardized_x ** 2           # standardized_xのすべての要素を二乗する
         ]).T                          # 作成した行列を転置する
    

def f(matrix_x, parameter):
    '''
        多項式回帰
        
        パラメーター
        ------------
        matrix_x  : 標準化後のxの行ベクトル
    parameter : パラメーターΦ、Φ1、Φ2の行ベクトル

    '''
    # matrix_xとparameterの積を返す
    return np.dot(matrix_x, parameter)

def MSE(matrix_x, y, parameter):
    '''
        平均二乗誤差を求める
        
        パラメーター
        ------------
        matrix_x  : xの行列
        y         : yの値
        parameter : パラメーターΦ0、Φ1、Φ2の行ベクトル
    '''
    return (1 / matrix_x.shape[0] # データのサイズで割る
            ) * np.sum(           # 二乗和誤差を求める
                (y - f(matrix_x,  # 学習データyと予測値の差
                       parameter
                       )
                 ) ** 2)          # 二乗する

def SGD(matrix_x, y):
    '''
        確率的勾配降下法で最適化を行う
        
        パラメーター
        ------------
        matrix_x : xの行列
        y        : yの値
    '''
    # パラメータΦ0、Φ1、Φ2を初期化
    parameter = np.random.rand(3)

    # 学習率を0.001に設定
    LNR = 1e-3
    #LNR = 1e-2
    #LNR = 1e-4
    #LNR = 1e-1

    # パラメーターの更新前後の平均二乗誤差の差を保持する変数
    difference = 1

    # 更新回数をカウントする変数を0で初期化
    count = 0

    # パラメーターの初期値を使用して平均二乗誤差を求める
    before = MSE(matrix_x, # 行列xの転置行列
                 y,        # yの値
                 parameter # パラメーターΦ0、Φ1、Φ2のベクトル
                 )
    
    # 平均二乗誤差の差分が0.01以下になるまでパラメータの更新を繰り返す
    while difference > 1e-2:
        # 学習データを並べ替えるためのランダムな順列を生成
        p = np.random.permutation(matrix_x.shape[0])

        # 学習データからxとyのペアをランダムに1セットずつ取り出す
        for tmp_x, tmp_y in zip(matrix_x[p,:], # xの行列からpの行を抽出
                                y[p]           # yの行ベクトルからpの成分を抽出
                                ):

            # 確率的勾配降下法でパラメーターを更新する
            parameter = parameter - LNR * (f(tmp_x,    # 抽出したx
                                             parameter # パラメーターΦ0、Φ1、Φ2
                                             ) - tmp_y # 抽出したy
                                           ) * tmp_x

        # パラメーター更新後における平均二乗誤差を求める
        after = MSE(matrix_x, # xの行列
                    y,        # yの値
                    parameter # サンプルで計算されたΦ0、Φ1、Φ2
                    )

        # パラメーター更新前後における平均二乗誤差の差を求める
        difference = before - after
        
        # パラメーター更新後の平均二乗誤差を更新前の値としてbeforeに代入
        before = after
        
        # カウンター変数の値を1増やす
        count += 1

        # 処理結果を出力
        log = '({}) parameter: {} error: {:.4f}'
        print(log.format(count, parameter, difference ))
        
    # 多項式回帰で求めたパラメーターΦ0、Φ1、Φ2の行ベクトルを戻り値として返す
    return parameter

######## 実行ブロック ######################################################
if __name__ == '__main__':
        
    # 学習データを読み込む
    data = np.loadtxt('access.csv',              # 読み込むファイル
                      dtype='int',               # データ型を指定
                      delimiter=',',             # 区切り文字を指定
                      skiprows=1                 # 1行目のタイトルを読み飛ばす
                      )
    x = data[:,0]                                # 1列目の成分をxに代入
    y = data[:,1]                                # 2列目の成分をyに代入
    standardized_x = standardize(x)              # xのすべての要素を標準化

    matrix_x = create_matrix(standardized_x)     # 標準化したxを行列にする

    # 確率的勾配降下法でパラメーターΦ0、Φ1の値を求める
    parameter = SGD(matrix_x, y)
            
    # x軸の値として-3から3までの等差数列を生成
    x_axis = np.linspace(start=-3,               # 数列の始点
                         stop=3,                 # 数列の終点
                         num=100                 # 等差数列の要素数
                         )
    
    plt.ylim(300, 2500)                          # y軸の範囲を設定

    # 標準化したxの値とyの値が交差するポイントをプロット
    plt.plot(standardized_x, y, 'o')
    # x軸の等差数列に対応するyをf()関数で求め、回帰直線をプロット
    plt.plot(x_axis,                             # x軸の値
             f(create_matrix(x_axis), parameter) # y軸の値
             )
    plt.grid(True)
    # グラフを表示
    plt.show()

