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

# Bài 2:
# Bài 2a:
import numpy as np
from sklearn.preprocessing import add_dummy_feature

np.random.seed(42)
m = 100  # số mẫu
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)

X_b = add_dummy_feature(X)
# code here
eta = 0.1 # learning rate
n_epochs = 1000
m = len(X_b) # Gán lại m cho chắc chắn (dù ở trên đã có)

theta = np.random.randn(2, 1) # Khởi tạo ngẫu nhiên theta

for epoch in range(n_epochs):
    # Tính đạo hàm và cập nhật theta
    gradients = 2/m * X_b.T @ (X_b @ theta - y)
    theta = theta - eta * gradients

print("Theta sau khi huấn luyện:\n", theta)

# Bài 2b:
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import add_dummy_feature
import numpy as np

X_new = np.array([[0], [2]])
X_new_b = add_dummy_feature(X_new)

def plot_gradient_descent(theta, eta):
    m = len(X_b)
    plt.plot(X, y, "b.")
    n_epochs = 1000
    n_shown = 20
    
    for epoch in range(n_epochs):
        if epoch < n_shown:
            y_predict = X_new_b @ theta
            color = mpl.colors.rgb2hex(plt.cm.OrRd(epoch / n_shown + 0.15))
            plt.plot(X_new, y_predict, linestyle="solid", color=color)
        gradients = 2/m * X_b.T @ (X_b @ theta - y)
        theta = theta - eta * gradients
        
    plt.xlabel("$x_1$")
    plt.axis([0, 2, 0, 15])
    plt.grid()
    plt.title(fr"$\eta = {eta}$")

np.random.seed(42)
# Khởi tạo theta chung để cả 3 biểu đồ đều xuất phát từ 1 điểm giống nhau
theta_initial = np.random.randn(2, 1)
plt.figure(figsize=(10, 4))
# Vẽ 3 subplot tương ứng với 3 giá trị eta
etas = [0.02, 0.1, 0.5]
for i, eta in enumerate(etas):
    plt.subplot(1, 3, i + 1)
    plot_gradient_descent(theta_initial, eta)

plt.tight_layout()
plt.show()

# Bài 2c:
'''
Dựa vào 3 đồ thị, ta thấy Learning Rate ($\eta$) ảnh hưởng trực tiếp đến tốc độ và khả năng hội tụ của thuật toán:
    - Với $\eta = 0.02$ (quá nhỏ): Các đường thẳng đỏ nhích từng bước rất ngắn. Mô hình cuối cùng cũng sẽ tìm được đáy, nhưng mất quá nhiều thời gian.
    Sau 20 bước, đường dự đoán vẫn còn cách rất xa đường tối ưu.
    - Với $\eta = 0.1$ (phù hợp): Thuật toán hội tụ nhanh chóng và ổn định. Chỉ sau một vài bước đầu tiên, các đường dự đoán đã bám rất sát vào dữ liệu.
    -Với $\eta = 0.5$ (quá lớn): Bước nhảy quá dài khiến mô hình nhảy vọt qua lại (overshooting) hai bên sườn dốc của hàm loss. 
    Kết quả là nó bị phân kỳ (diverge), các đường dự đoán văng ra khỏi phạm vi dữ liệu và không bao giờ hội tụ được.
'''