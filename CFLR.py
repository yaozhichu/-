import numpy as np
import matplotlib.pyplot as plt
import os

file_path = os.path.join(os.path.dirname(__file__), 'x-y.txt')
with open(file_path, 'r') as file:
    x = np.array(list(map(float, file.readline().split())))
    y = np.array(list(map(float, file.readline().split())))
# 构造X矩阵，第一列为x值，第二列为全1（用于截距b的计算）
X = np.vstack([x, np.ones(len(x))]).T
# 线性回归显式解公式，求伪逆矩阵函数（考虑矩阵不可逆）
parameter = np.linalg.pinv(X.T @ X) @ X.T @ y
w, b =parameter

print(f"w为 {w:.3f}")
print(f"b为 {b:.3f}")

plt.scatter(x, y, color='blue', label='Data_Points')
plt.plot(x, w * x + b, color='green', label=f'Line_Ression: y = {w:.2f}x + {b:.2f}')

plt.xlabel('x')
plt.ylabel('y')
plt.title('CFLR')
plt.legend()
plt.show()
