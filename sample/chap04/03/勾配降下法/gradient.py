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

def f(standardized_x, parameter_0, parameter_1):
    '''
        回帰直線を求める関数
        
        パラメーター
        ------------
        standardized_x : 標準化後のx
        parameter_0    : パラメーターΦ0
        parameter_1    : パラメーターΦ1
    '''
    # f(Φ)=Φ_0 + Φ_1・xを戻り値として返す
    return parameter_0 + parameter_1 * standardized_x

def objective_f(standardized_x, y, parameter_0, parameter_1):
    '''
        目的関数
        
        パラメーター
        ------------
        standardized_x : 標準化後のx
        y              : yの値
        parameter_0    : パラメーターΦ0
        parameter_1    : パラメーターΦ1
    '''
    return 0.5 * np.sum((y - f(standardized_x,
                               parameter_0,
                               parameter_1)) ** 2)


def gradient_method(standardized_x, y):
    '''
        勾配法で最適化を行う
        
        パラメーター
        ------------
        standardized_x : 標準化後のx
        y              : yの値
    '''
    # パラメーターを初期化
    parameter_0 = np.random.rand()
    parameter_1 = np.random.rand()

    LNR = 1e-3    # 学習率を0.001にセット
    #LNR = 1e-2
    #LNR = 1e-4

    # パラメーターの更新前後の目的関数の値の差を保持する変数を1で初期化
    difference = 1

    # 更新回数をカウントする変数を0で初期化
    count = 0

    # パラメーターの初期値を使用した目的関数の値をbeforeに代入
    before = objective_f(standardized_x, y,       # x、yのデータ
                         parameter_0, parameter_1 # パラメーターΦ0、Φ1
                         )

    # 誤差の差分が0.01以下になるまでパラメータ更新を繰り返す
    while difference > 1e-2: # 0.01
        # Φ0を更新する
        tmp_parameter_0 = parameter_0 - LNR * np.sum((f(standardized_x,
                                                        parameter_0,
                                                        parameter_1) - y))
        # Φ1を更新する
        tmp_parameter_1 = parameter_1 - LNR * np.sum((f(standardized_x,
                                                        parameter_0,
                                                        parameter_1) - y)
                                                     * standardized_x)

        # 更新後のΦ0、Φ1の値をパラメーター用の変数に代入
        parameter_0 = tmp_parameter_0
        parameter_1 = tmp_parameter_1

        # 更新後のΦ0、Φ1の値で目的関数の値を求める
        after = objective_f(standardized_x, y,       # x、yのデータ
                            parameter_0, parameter_1 # 更新後のΦ0、Φ1
                            )
        
        # パラメーター更新前後における目的関数の値の差を求める
        difference = before - after
        # パラメーター更新後の目的関数の値を更新前の値としてbeforeに代入
        before = after        

        # カウンター変数の値を1増やす
        count += 1
        # 処理結果を出力
        log = '({}) Φ0: {:.3f} Φ1: {:.3f} error: {:.4f}'
        print(log.format(count, parameter_0, parameter_1, difference))

    # 勾配法で求めたパラメーターΦ0、Φ1の値を戻り値として返す
    return parameter_0, parameter_1
    
   
######## 実行ブロック ######################################################
if __name__ == '__main__':

    # 学習データを読み込む
    data = np.loadtxt(fname='access.csv', # 読み込むファイル
                      dtype='int',        # データ型を指定
                      delimiter=',',      # 区切り文字を指定
                      skiprows=1          # 1行目のタイトルを読み飛ばす
                      )
    x = data[:,0]                         # 1列目の成分をxに代入
    y = data[:,1]                         # 2列目の成分をyに代入
    standardized_x = standardize(x)       # xのすべての要素を標準化

    # 勾配法でパラメーターΦ0、Φ1の値を求める
    parameter_0, parameter_1 = gradient_method(standardized_x, y)

    # x軸の値として-3から3までの等差数列を生成
    x_axis = np.linspace(start=-3,       # 数列の始点
                         stop=3,         # 数列の終点
                         num=100         # 等差数列の要素数
                         )
    
    # 標準化したxの値とyの値が交差するポイントをプロット
    plt.plot(standardized_x, y, 'o')
    # x軸の等差数列に対応するyをf()関数で求め、回帰直線をプロット
    plt.plot(x_axis, f(x_axis, parameter_0, parameter_1))
    plt.grid(True)
    # グラフを表示
    plt.show()
