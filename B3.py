import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import add_dummy_feature
from sklearn.linear_model import SGDRegressor

np.random.seed(42)
m = 100
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)
X_b = add_dummy_feature(X)

# BÀI 3a: STOCHASTIC GRADIENT DESCENT (SGD) THỦ CÔNG
print("--- Bài 3a: SGD Thủ công ---")
n_epochs = 50
t0, t1 = 5, 50

# Hàm giảm learning rate theo thời gian
def learning_schedule(t):
    return t0 / (t + t1)
np.random.seed(42)
theta = np.random.randn(2, 1)
for epoch in range(n_epochs):
    for iteration in range(m):
        random_index = np.random.randint(m)
        
        xi = X_b[random_index : random_index + 1]
        yi = y[random_index : random_index + 1]

        gradients = 2 * xi.T @ (xi @ theta - yi)

        eta = learning_schedule(epoch * m + iteration)

        theta = theta - eta * gradients

print("SGD theta =\n", theta)

# BÀI 3b: SO SÁNH VỚI SGDRegressor CỦA SKLEARN
print("\n--- Bài 3b: SGDRegressor Sklearn ---")
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, penalty=None, 
                       eta0=0.01, n_iter_no_change=100, random_state=42)

sgd_reg.fit(X, y.ravel())

print("intercept:", sgd_reg.intercept_)
print("coef:", sgd_reg.coef_)

# BÀI 3c: MINI-BATCH GRADIENT DESCENT
print("\n--- Bài 3c: Mini-batch GD ---")
n_epochs = 50
batch_size = 20
t0, t1 = 200, 1000

def learning_schedule_mb(t):
    return t0 / (t + t1)

np.random.seed(42)
theta = np.random.randn(2, 1)
t = 0  # Biến đếm tổng số lần cập nhật

for epoch in range(n_epochs):
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    
    for i in range(0, m, batch_size):
        t += 1
        X_batch = X_b_shuffled[i : i + batch_size]
        y_batch = y_shuffled[i : i + batch_size]
        
        gradients = 2 / batch_size * X_batch.T @ (X_batch @ theta - y_batch)
        eta = learning_schedule_mb(t)
        theta = theta - eta * gradients

print("Mini-batch GD theta =\n", theta)