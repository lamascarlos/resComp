"""Linear readout: ridge regression on z-scored features with an unpenalized intercept, solved by SVD so a whole
grid of ridge parameters and many targets cost one decomposition."""
import numpy as np

ALPHAS = np.logspace(-6, 4, 11)


class Ridge:
    """Multi-target ridge regression.

    Features are z-scored with training statistics; features whose training std is <= min_std are dropped
    (round-off-level variation would otherwise be blown up to unit variance). The intercept is the training mean
    of the target (ridge on centred data), so it is not penalized.
    """

    def __init__(self, min_std: float = 1e-9):
        self.min_std = min_std

    def fit_svd(self, X: np.ndarray, Y: np.ndarray):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        std = X.std(axis=0)
        self.keep_ = std > self.min_std
        self.mean_ = X[:, self.keep_].mean(axis=0)
        self.scale_ = std[self.keep_]
        self.y_mean_ = Y.mean(axis=0)
        Z = (X[:, self.keep_] - self.mean_) / self.scale_
        self.U_, self.s_, Vt = np.linalg.svd(Z, full_matrices=False)
        self.V_ = Vt.T
        self.UtY_ = self.U_.T @ (Y - self.y_mean_)
        return self

    def _standardize(self, X):
        return (np.asarray(X, dtype=float)[:, self.keep_] - self.mean_) / self.scale_

    def weights(self, alpha: float) -> np.ndarray:
        """(n_kept_features, n_targets) weights for ridge parameter alpha (on standardized features)."""
        d = self.s_ / (self.s_ ** 2 + alpha)
        return self.V_ @ (d[:, None] * self.UtY_)

    def predict(self, X: np.ndarray, alpha) -> np.ndarray:
        """Predictions for one alpha (scalar) or one alpha per target (array)."""
        B = self._standardize(X) @ self.V_                          # (n, r)
        alpha = np.asarray(alpha, dtype=float)
        if alpha.ndim == 0:
            d = self.s_ / (self.s_ ** 2 + alpha)
            return B @ (d[:, None] * self.UtY_) + self.y_mean_
        d = self.s_[:, None] / (self.s_[:, None] ** 2 + alpha[None, :])   # (r, m)
        return B @ (d * self.UtY_) + self.y_mean_


def fit_predict(X, Y, train, val, test, alphas=ALPHAS, score="mse", min_std=1e-9):
    """Fit on `train`, choose alpha per target on `val`, refit on train + val, predict `test`.

    score: "mse" (regression) or "accuracy" (targets are +-1 labels, prediction sign).
    Returns (test predictions (n_test, m), chosen alphas (m,), validation predictions (n_val, m))."""
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    model = Ridge(min_std).fit_svd(X[train], Y[train])
    B = model._standardize(X[val]) @ model.V_
    losses = []
    for a in alphas:
        d = model.s_ / (model.s_ ** 2 + a)
        pred = B @ (d[:, None] * model.UtY_) + model.y_mean_
        if score == "accuracy":
            losses.append(-np.mean(np.sign(pred) == Y[val], axis=0))
        else:
            losses.append(np.mean((pred - Y[val]) ** 2, axis=0))
    best = np.asarray(alphas)[np.argmin(np.array(losses), axis=0)]
    val_pred = model.predict(X[val], best)
    trval = np.concatenate([train, val])
    final = Ridge(min_std).fit_svd(X[trval], Y[trval])
    return final.predict(X[test], best), best, val_pred
