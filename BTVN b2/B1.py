import sys
assert sys.version_info >= (3, 7)

import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

np.random.seed(42)

# Bài 1:
np.random.seed(42)
m = 100  # số mẫu
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)

# Vẽ dữ liệu
plt.figure(figsize=(6, 4))
plt.plot(X, y, "b.")
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)
plt.axis([0, 2, 0, 15])
plt.grid()
plt.title("Dữ liệu giả lập")
plt.show()

# Bài 1a:
from sklearn.preprocessing import add_dummy_feature
X_b = add_dummy_feature(X)  # thêm cột bias
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) #Tính theta_best bằng Normal Equation
print("Theta best:", theta_best)

# Bài 1b:
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression() # Khởi tạo mô hình
lin_reg.fit(X, y) # Huấn luyện mô hình
print('Intercept:', lin_reg.intercept_)
print('Coef:', lin_reg.coef_)

# Bài 1c:
import matplotlib.pyplot as plt
X_new = np.array([[0], [2]])  # Tạo một mảng mới để dự đoán
y_predict = lin_reg.predict(X_new)  # Dự đoán giá trị y cho X_new
plt.figure(figsize=(6, 4)) # Vẽ biểu đồ
plt.plot(X_new, y_predict, "r-", linewidth=2, label="Dự đoán")
plt.plot(X, y, "b.", label="DATA")
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)
plt.axis([0, 2, 0, 15])
plt.grid()
plt.legend(loc="upper left")
plt.show()
