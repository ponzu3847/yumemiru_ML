import numpy as np              # numpyのインポート

a = np.array([[2, 3, 4],
              [5, 6, 7],
              [8, 9, 1]])

zero = np.zeros((3, 3))

unit = np.identity(3)

print(zero)
print(unit)
print(a * zero)
print(np.dot(a, unit))

a = np.array([[1, 2, 3],
              [4, 5, 6]])
print(a)
print(a.T) 

a = np.array([[1, 2],       # 2×2の行列を作成
              [3, 4]]
            )
inv = np.linalg.inv(a)      # 逆行列を求める
print(inv)
print(np.dot(a, inv))    # AB = E、BA = Eとなるのか確かめる
