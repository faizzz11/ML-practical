# Step 1: Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm

# Step 2: Load the Iris dataset (only first 2 features for 2D plotting)
iris_data = datasets.load_iris()
features = iris_data.data[:, :2]    # Sepal length & Sepal width
labels = iris_data.target

# Step 3: Initialize and train SVM model
# Using linear kernel, regularization parameter C = 1.0
model = svm.SVC(kernel='linear', C=1.0).fit(features, labels)

# Step 4: Create meshgrid for decision boundary plotting
x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1
step = 0.02   # step size for the grid
xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                     np.arange(y_min, y_max, step))

# Step 5: Predict class for each point in the grid
predictions = model.predict(np.c_[xx.ravel(), yy.ravel()])
predictions = predictions.reshape(xx.shape)

# Step 6: Plot decision boundaries and data points
plt.figure(figsize=(7, 5))
plt.contourf(xx, yy, predictions, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(features[:, 0], features[:, 1], c=labels, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.title('SVM Decision Boundary - Linear Kernel [Faiz Moulavi 45]')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
