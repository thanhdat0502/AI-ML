import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)

print("Feature names:", list(iris.data.columns))
print("Target names:", list(iris.target_names))
print("Shape:", iris.data.shape)

# Bài 4a:
from sklearn.linear_model import LogisticRegression
X_iris = iris.data[["petal width (cm)"]].values  
y_iris = (iris.target == 2).astype(int)  
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_iris, y_iris)

# Bài 4b:
X_new_iris = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new_iris)
plt.figure(figsize=(8, 4))
plt.plot(X_new_iris, y_proba[:, 1], "g-", linewidth=2, label="Iris virginica")
plt.plot(X_new_iris, y_proba[:, 0], "b--", linewidth=2, label="Not Iris virginica")

plt.xlabel("Petal width (cm)")
plt.ylabel("Probability")
plt.legend(loc="center left")
plt.grid()
plt.show()


# Bài 4c:
X_test = np.array([[1.7], [2.0]])
y_pred = log_reg.predict(X_test)
y_prob = log_reg.predict_proba(X_test)[:, 1]
print(f"Dự đoán cho petal_width = 1.7: Class = {y_pred[0]} (Xác suất là Virginica: {y_prob[0]*100:.2f}%)")
print(f"Dự đoán cho petal_width = 2.0: Class = {y_pred[1]} (Xác suất là Virginica: {y_prob[1]*100:.2f}%)")