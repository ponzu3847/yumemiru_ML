import numpy as np
import matplotlib.pyplot as plt

def distribution(x):
     '''
        データの分布を表す関数
        
        パラメーター
        ------------
        x : 学習データ
    '''
     return -0.09 * (x + x**2)

def f(matrix_x, parameter):
    '''
        回帰式
        
        パラメーター
        ------------
        matrix_x  : xの行列
        parameter : パラメーターθの行ベクトル
    '''
    # matrix_xとparameterの積を返す
    return np.dot(matrix_x, parameter)

def objective_f(matrix_x, y, parameter):
    '''
        目的関数
        
        パラメーター
        ------------
        matrix_x  : xの行列
        y         : yの値
        parameter : パラメーターθの行ベクトル
    '''
    return 0.5 * np.sum((y - f(matrix_x,
                               parameter)
                         ) ** 2)

def create_matrix(x):
    '''
        6次の多項式で学習データの行列を作る      
    '''
    return np.vstack([
        np.ones(x.size), # バイアス項を加える
        x, x ** 2, x ** 3, x ** 4, x ** 5, x ** 6
        ]).T
                         
# 学習率を0.0001に設定
LNR = 1e-4

def parameter_update(matrix_x, y):
    '''
        正則化を適用せずに学習を繰り返す
        
        パラメーター
        ------------
        matrix_x : xの行列
        y        : yの値
    '''
    # パラメーターを初期化
    parameter = np.random.randn(matrix_x.shape[1])
    # パラメーターの更新前後の目的関数の値の差を保持する変数を1で初期化
    difference  = 1
    
    # パラメーターの初期値を使用して目的関数を実行し、戻り値をbeforeに代入
    before = objective_f(matrix_x,          # xの行列
                         y,                 # yの値
                         parameter          # パラメーターθ0、θ1、θ2のベクトル
                        )
    
    # 誤差の差分が0.000001以下になるまでパラメータの更新を繰り返す
    while difference > 1e-6:
        # θを更新する
        parameter = parameter - LNR * np.dot(f(matrix_x,
                                               parameter) - y,
                                             matrix_x)
        # 更新後のθの値で目的関数の値を求める
        after = objective_f(matrix_x,       # xの行列
                            y,              # yの値
                            parameter       # パラメーターθのベクトル
                            )

        # パラメーター更新前後における目的関数の値の差を求める
        difference  = before - after
        # パラメーター更新後の目的関数の値を更新前の値としてbeforeに代入
        before = after 
     
    # 学習によって求めたパラメーターθの行ベクトルを戻り値として返す
    return parameter

def parameter_update_regularization(matrix_x, y,reg=1):
    '''
        正則化を適用して学習を繰り返す
        
        パラメーター
        ------------
        matrix_x : xの行列
        y        : yの値
    '''
    # パラメーターを初期化
    parameter = np.random.randn(matrix_x.shape[1])
    # パラメーターの更新前後の目的関数の値の差を保持する変数を1で初期化
    difference  = 1
                         
    # 正則化定数
    REG = reg

    # パラメーターの初期値を使用して目的関数を実行し、戻り値をbeforeに代入
    before = objective_f(matrix_x,          # xの行列
                         y,                 # yの値
                         parameter          # パラメーターθ0、θ1、θ2のベクトル
                        )

    # 誤差の差分が0.000001以下になるまでパラメータの更新を繰り返す
    while difference > 1e-6:
        # 正則化項の計算
        reg_term = REG * np.hstack(         # 行方向に連結
                             [0,            # バイアス項を追加
                              parameter[1:]]# バイアス項を含まないパラメーターθ
                            )
        # 正則化を行ってパラメーターθを更新する
        parameter = parameter - LNR * (np.dot(f(matrix_x,
                                                parameter) - y,
                                                matrix_x) + reg_term)
                                    
        # 更新後のθの値で目的関数の値を求める
        after = objective_f(matrix_x,       # xの行列
                            y,              # yの値
                            parameter       # パラメーターθのベクトル
                            )

        # パラメーター更新前後における目的関数の値の差を求める
        difference  = before - after
        # パラメーター更新後の目的関数の値を更新前の値としてbeforeに代入
        before = after 

    # 学習によって求めたパラメーターθの行ベクトルを戻り値として返す
    return parameter


######## 実行ブロック ######################################################
if __name__ == '__main__':

    # 正則化定数の値を取得
    REG = float(input('正則化定数->'))
    
    # xの値を等間隔に20個生成する
    x = np.linspace(start=-3,               # 数列の始点
                    stop=3,                 # 数列の終点
                    num=20                  # 等差数列の要素数
                    )
					
    # 関数が出力値したy値にバラつきを加える
    np.random.seed(0)                      # 毎回、同じ乱数を発生させる
    y = distribution(x) + np.random.randn(x.size)*0.1

    x_mean = x.mean()                       # xの平均値
    x_std = x.std()                         # xの標準偏差
    standardized_x = (x - x_mean)/x_std     # xのすべての成分を標準化する
    matrix_x = create_matrix(standardized_x)# 標準化したxを行列にする

    # 正則化なしでパラメーターθを学習する
    no_regularization = parameter_update(matrix_x, y)
    # 正則化を適用してパラメーターΦを学習する
    do_regularization = parameter_update_regularization(matrix_x, y, reg=REG)

    # x軸の値として-3から3までの等差数列を生成
    x0 = np.linspace(start=-3,              # 数列の始点
                     stop=3,                # 数列の終点
                     num=100                # 等差数列の要素数
                     )
    # スケールを合わせるためxの平均、標準偏差で標準化する
    std_x0 = (x0 - x_mean)/x_std      
    # std_x0を行列にする
    matrix_x0 = create_matrix(std_x0)

    # 学習データをプロット
    plt.plot(standardized_x, y, '^')
	
    # 正則化なしパラメーターを用いた曲線をプロット
    plt.plot(std_x0,                        # 標準化後のx0
             f(matrix_x0,                   # x0の行列
               no_regularization            # 正則化なしのパラメーター
               ),
             linestyle='solid')             # 実線を描画
    
    # 正則化適用のパラメーターを用いた曲線をプロット
    plt.plot(std_x0,                        # 標準化後のx0
             f(matrix_x0,                   # x0の行列
               do_regularization            # 正則化ありのパラメーター
               ),
             linestyle='dashed')            # 破線を描画

    plt.grid(True)
    plt.show()

