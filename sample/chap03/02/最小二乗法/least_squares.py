import csv                           # csvモジュールをインポート
import math                          # mathモジュールをインポート


def regression(file):
    '''
        最小2乗法で回帰係数と切片を求める
        
        パラメーター
        ------------
        file : str
            読み込みを行うcsvファイル名
    '''
    i = 0                            # カウンター変数を初期化
    a = 0.0                          # 回帰直線の傾き(回帰係数)の初期化
    b = 0.0                          # y切片の初期化

    sum_x  = 0.0                     # xの和
    sum_y  = 0.0                     # yの和
    sum_xy = 0.0                     # xとyの積和
    sum_x2 = 0.0                     # xの平方和

    with open(file, 'r') as f:       # ファイルを読み取りモードで開く
        reader = csv.reader(f)
        next(reader)                 # ヘッダーを読み飛ばす

        for row in reader:           # 1行データをリストとして取得
            x = float(row[0])        # xの値を取得
            y = float(row[1])        # yの値を取得
            sum_x  += x              # x値を加算
            sum_y  += y              # y値を加算
            sum_xy += x * y          # xyの積
            sum_x2 += math.pow(x, 2) # xの平方和
            i += 1                   # カウンターに1加算


    # a（傾き）を求める
    a = (i * sum_xy - sum_x * sum_y) / (i * sum_x2 - math.pow(sum_x, 2))
    # b(切片)を求める
    b = (sum_x2 * sum_y - sum_xy * sum_x) / (i * sum_x2 - math.pow(sum_x, 2))
    print('傾き = ' , a)
    print('切片 = ' , b)

######## 実行ブロック ######################################################
if __name__ == '__main__':

    regression('data.csv')

