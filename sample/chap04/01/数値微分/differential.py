import numpy as np                     # numpyのインポート
import matplotlib.pyplot as plt        # matplotlib.pylabをpltとして使用する

def differential(f,
                 x
                 ):
    '''
    数値微分を行う高階関数
    
    パラメーター
    ----------
    f : 関数オブジェクト
        数値微分に用いる関数
    x : int
        f(x)のxの値
    
    戻り値
    ----------
        数値微分の結果
    '''
    h = 1e-4                           # hの値を0.0001にする
    return (
        f(x+h) - f(x-h)) / (2*h)       # 数値微分して変化量を戻り値として返す

def function(x
             ):
    '''
    数値微分で使用する関数
	
    パラメーター
    ----------
    x : int
        f(x)のxの値
	 
    戻り値
    ----------
    float
        yの値
    '''
    return 0.01*x**2 + 0.1*x

def draw_line(f, x):
    '''
    数値微分の値を傾きとする直線をプロットするラムダ式を生成する関数
	differential()を実行する
	
    パラメーター
    ----------
    f : 関数オブジェクト
        数値微分に用いる関数を取得
    x : int
        f(x)のxの値

    戻り値
    ----------
    lambdaオブジェクト
        数値微分の値を傾きとする直線をプロットするためのラムダ式
    '''
    dff = differential(f, x)     # differential()で数値微分を行い、変化量を取得
    print(dff)                   # 変化量(直線の傾き）をを出力
    y = f(x) - dff * x           # f(x)にxを代入したyと変化量から求めたyとの差
    return lambda n: dff*n + y   # 引数をtで受け取るラムダ式
                                 # 「変化量 × x軸の値(t) + f(x)との誤差」
                                 # f(x)との誤差を加えることで直線が接するようにする


######## 実行ブロック ######################################################
if __name__ == '__main__':
    
    x = np.arange(0.0, 20.0, 0.1)    # 0.0から20.0までの0.1刻みの配列(ndarray)を生成
    y = function(x)                  # 関数f(x)に配列xを代入し、0.0から20.0までのy値のリストを取得
    plt.xlabel("x")                  # x軸のラベルを設定
    plt.ylabel("f(x)")               # y軸のラベルを設定

    tf = draw_line(function, 5)      # 数値微分の値を傾きとする直線のラムダ式を取得
    y2 = tf(x)                       # 取得したラムダ式で0.0から20.0までの0.1刻みのyの値を取得
    plt.plot(x, y)                   # f(x)をプロット
    plt.plot(x, y2)                  # 数値微分の値を傾きとする直線をプロット
    plt.show()                       # グラフを描画
