import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample


# ----------------------------
# Ridge regression helpers
# ----------------------------
def polynomial_features(x, degree):
    """Return polynomial design matrix without intercept."""
    x = np.asarray(x).ravel()
    return np.column_stack([x**k for k in range(1, degree + 1)])


def ridge_parameters(X, y, lam):
    """Closed-form ridge regression solution."""
    XT_X = X.T @ X
    p = XT_X.shape[0]
    return np.linalg.solve(XT_X + lam * np.eye(p), X.T @ y)

