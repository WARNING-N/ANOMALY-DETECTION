
import numpy as np
from pyod.utils.data import generate_data
from sklearn.metrics import balanced_accuracy_score

contamination = 0.1

X_train, _, y_train, _ = generate_data(
    n_train=1000,
    n_test=0,
    n_features=1,
    contamination=contamination
)

X = X_train.ravel() # make 1D

mu = np.mean(X)
sigma = np.std(X)


z_scores = np.abs((X - mu) / sigma)

threshold = np.quantile(z_scores, 1 - contamination)

y_pred = (z_scores > threshold).astype(int)

ba = balanced_accuracy_score(y_train, y_pred)

print("Z-score threshold:", threshold)
print("Balanced accuracy:", ba)
