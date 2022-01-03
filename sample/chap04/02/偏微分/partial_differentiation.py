import numpy as np               # numpyのインポート
import matplotlib.pyplot as plt  # matplotlib.pylabをpltとして使用する

def differential(f,              # 数値微分に用いる関数を取得
                 x               # f(x)のxの値
                 ):
    '''
    数値微分を行う高階関数
    
    戻り値
    ----------
        数値微分の結果
    '''
    h = 1e-4                     # hの値を0.0001にする
    return (
        f(x+h) - f(x-h)) / (2*h) # 数値微分して変化量を戻り値として返す

def function_1(x1                # int: f(x1, x2)のx1の値
               ):
    '''
    x1に対する偏微分で使用する関数

    戻り値
    ----------
    float
        x1=3、x2=4のときのx1に対する偏微分の結果
    '''
    return x1*x1 + 4.0**2.0

def function_2(x2                # int: f(x1, x2)のx2の値
               ):
    '''
    x2に対する偏微分で使用する関数

    戻り値
    ----------
    float
        x1=3、x2=4のときのx2に対する偏微分の結果
    '''
    return x2*x2 + 3.0**2.0


######## 実行ブロック ######################################################
if __name__ == '__main__':

    print("x1に対する偏微分: ",
          differential(function_1, 3.0)
          )
    
    print("x2に対する偏微分: ",
          differential(function_2, 4.0)
          )


