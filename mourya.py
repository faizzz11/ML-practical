import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Generate linearly separable dataset
X, y = make_classification(
    n_samples=200, n_features=2, n_redundant=0, n_informative=2,
    n_clusters_per_class=1, class_sep=2.0, random_state=42
)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train SVM with Linear Kernel
svm_clf = SVC(kernel="linear", C=1)
svm_clf.fit(X_train, y_train)

# Predict on test set
y_pred = svm_clf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# Function to plot linear decision boundary
def plot_linear_boundary(clf, X, y):
    # plot points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors="k")
    
    # get coefficients
    w = clf.coef_[0]
    b = clf.intercept_[0]
    
    # plot line
    x_vals = np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100)
    y_vals = -(w[0] * x_vals + b) / w[1]
    plt.plot(x_vals, y_vals, 'k--')
    
    plt.title("SVM with Linear Kernel [Aaditya Mourya 46]")
    plt.show()
    
# Plot decision boundary
plot_linear_boundary(svm_clf, X, y)



#2 partttttttttttt


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Generate a non-linear dataset (two interleaving half circles)
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train SVM with RBF kernel (kernel trick)
svm_clf = SVC(kernel="rbf", gamma=1, C=1)
svm_clf.fit(X_train, y_train)

# Predict on test set
y_pred = svm_clf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# Visualization function for decision boundary
def plot_decision_boundary(clf, X, y):
    # create grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    
    # predict for each grid point
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors="k")
    plt.title("SVM with RBF Kernel (Kernel Trick) by Aaditya Mourya 46")
    plt.show()

# Plot decision boundary
plot_decision_boundary(svm_clf, X, y)
