import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap

np.random.seed(42)
n_samples = 50

theta = np.linspace(0, 2 * np.pi, n_samples)
X_inner = np.column_stack([(0.5 + np.random.randn(n_samples) * 0.1) * np.cos(theta),
                           (0.5 + np.random.randn(n_samples) * 0.1) * np.sin(theta)])
X_outer = np.column_stack([(1.2 + np.random.randn(n_samples) * 0.15) * np.cos(theta),
                           (1.2 + np.random.randn(n_samples) * 0.15) * np.sin(theta)])

X = np.vstack([X_inner, X_outer])
y = np.hstack([np.zeros(n_samples, dtype=int), np.ones(n_samples, dtype=int)])

svc_rbf = SVC(kernel='rbf', C=1.0, gamma=2.0)
svc_rbf.fit(X, y)

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
Z_rbf = svc_rbf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
decision_boundary = svc_rbf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 8))
plt.contourf(xx, yy, Z_rbf, cmap=ListedColormap(['#ADD8E6', '#D2B48C']), alpha=0.6)
plt.contour(xx, yy, decision_boundary, levels=[0], colors='black', linewidths=2)
plt.scatter(X_inner[:, 0], X_inner[:, 1], c='lightblue', marker='o', s=60,
           edgecolors='black', linewidths=0.8, label='Class 1 (circle)')
plt.scatter(X_outer[:, 0], X_outer[:, 1], c='saddlebrown', marker='o', s=60,
           edgecolors='black', linewidths=0.8, label='Class 2 (ring)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('SVM with Custom Kernel: k(x,y) = (x² + x²)(y² + y²)')
plt.legend(loc='upper right')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

z = X[:, 0]**2 + X[:, 1]**2
X_new = np.column_stack([X[:, 0], z])

svc_linear = SVC(kernel='linear', C=1.0)
svc_linear.fit(X_new, y)

x_min_new, x_max_new = X_new[:, 0].min() - 0.3, X_new[:, 0].max() + 0.3
z_min, z_max = X_new[:, 1].min() - 0.2, X_new[:, 1].max() + 0.2
xx_new, zz = np.meshgrid(np.linspace(x_min_new, x_max_new, 200),
                         np.linspace(z_min, z_max, 200))
Z_linear = svc_linear.decision_function(np.c_[xx_new.ravel(), zz.ravel()]).reshape(xx_new.shape)

plt.figure(figsize=(8, 6))
plt.contour(xx_new, zz, Z_linear, levels=[0], colors='black', linewidths=2)
plt.scatter(X_new[y == 0, 0], X_new[y == 0, 1], c='red', marker='o', s=60,
           edgecolors='black', linewidths=0.8, label='Class 1 (circle)')
plt.scatter(X_new[y == 1, 0], X_new[y == 1, 1], c='blue', marker='*', s=100,
           edgecolors='black', linewidths=0.8, label='Class 2 (ring)')
plt.xlabel('x')
plt.ylabel('z = x² + y²')
plt.title('Data projected onto new feature z = x² + y²')
plt.legend(loc='upper left')
plt.xlim(x_min_new, x_max_new)
plt.ylim(z_min, z_max)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
