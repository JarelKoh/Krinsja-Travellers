import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

def ridge_mse_heatmap(degrees, lambdas, X, y, test_size=0.3, seed=42):
    """Compute MSE grid for Ridge regression."""
    np.random.seed(seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    mse_scores = np.zeros((len(degrees), len(lambdas)))
    for i, d in enumerate(degrees):
        for j, lmb in enumerate(lambdas):
            model = make_pipeline(PolynomialFeatures(d), Ridge(alpha=lmb, random_state=seed))
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse_scores[i, j] = mean_squared_error(y_test, y_pred)
    return mse_scores
