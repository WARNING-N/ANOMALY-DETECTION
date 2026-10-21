
import numpy as np
from sklearn.metrics import balanced_accuracy_score

np.random.seed(23)

mu = np.array([2.0, -1.0])

Sigma = np.array([
    [4.0, 1.2],
    [1.2, 2.0]
])

# Sigma = L @ L.T
L = np.linalg.cholesky(Sigma)

n_inliers = 900
n_outliers = 100


X = np.random.randn(n_inliers, 2)

Y_inliers = X @ L.T + mu

X_outliers = np.random.randn(n_outliers, 2)
Y_outliers = X_outliers @ L.T + mu + np.array([11.0, 11.0])

Y = np.vstack([Y_inliers, Y_outliers])
y_true = np.concatenate([
    np.zeros(n_inliers, dtype=int),
    np.ones(n_outliers, dtype=int)
])

diff = Y - mu
z_scores = np.sum(
    diff * np.linalg.solve(Sigma, diff.T).T,
    axis=1
)

contamination = 0.1
threshold = np.quantile(z_scores, 1 - contamination)

y_pred = (z_scores > threshold).astype(int)

ba = balanced_accuracy_score(y_true, y_pred)

print("Threshold:", threshold)
print("Balanced accuracy:", ba)
