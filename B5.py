import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
# BÀI 5a:
X_softmax = iris["data"][["petal length (cm)", "petal width (cm)"]].values
y_softmax = iris["target"].values

softmax_reg = LogisticRegression(solver="lbfgs", C=10, random_state=42)
softmax_reg.fit(X_softmax, y_softmax)



# BÀI 5b: 
sample = np.array([[5, 2]])
print("Predicted class:", softmax_reg.predict(sample))
print("Probabilities:", softmax_reg.predict_proba(sample))



# BÀI 5c:
custom_cmap = ListedColormap(["#fafab0", "#9898ff", "#a0faa0"])
x0_grid = np.linspace(0, 7, 500)
x1_grid = np.linspace(0, 3.5, 200)
x0_mesh, x1_mesh = np.meshgrid(x0_grid, x1_grid)
X_grid = np.c_[x0_mesh.ravel(), x1_mesh.ravel()]
y_pred_grid = softmax_reg.predict(X_grid).reshape(x0_mesh.shape)
plt.figure(figsize=(8, 6))
plt.contourf(x0_mesh, x1_mesh, y_pred_grid, cmap=custom_cmap, alpha=0.8)
plt.scatter(X_softmax[:, 0], X_softmax[:, 1], c=y_softmax, cmap=plt.cm.brg, edgecolors="k")
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.title("Softmax Regression Decision Boundary")
plt.show()