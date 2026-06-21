import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer
from numba import njit

@njit
def concordance_correlation_numba(y_true, y_pred):
    n = y_true.size
    mt = 0.0
    mp = 0.0
    for i in range(n):
        mt += y_true[i]
        mp += y_pred[i]
    mt /= n
    mp /= n

    vt = 0.0
    vp = 0.0
    cov = 0.0
    for i in range(n):
        xt = y_true[i] - mt
        xp = y_pred[i] - mp
        vt += xt * xt
        vp += xp * xp
        cov += xt * xp
    vt /= n
    vp /= n
    cov /= n

    denom_r = (vt * vp) ** 0.5
    if denom_r == 0.0:
        return np.nan
    r = cov / denom_r
    return (2.0 * r * (vt ** 0.5) * (vp ** 0.5)) / (vt + vp + (mt - mp) ** 2)

def concordance_correlation(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cor = np.corrcoef(y_true, y_pred, bias=True)[0, 1]

    std_true = y_true.std(ddof=0)
    std_pred = y_pred.std(ddof=0)

    mean_true = y_true.mean()
    mean_pred = y_pred.mean()

    return 2 * cor * std_true * std_pred / (
        std_true**2 + std_pred**2 + (mean_true - mean_pred)**2
    )

def multi_concordance_correlation(y_true, y_pred):
    my_true = np.asarray(y_true)
    my_pred = np.asarray(y_pred)
    n = my_true.shape[1]
    assert y_true.shape == y_pred.shape
    sum_ = 0
    for index in range(n):
        y_true = my_true[:,index]
        y_pred = my_pred[:, index]
        cor = np.corrcoef(y_true, y_pred, bias=True)[0, 1]

        std_true = y_true.std(ddof=0)
        std_pred = y_pred.std(ddof=0)

        mean_true = y_true.mean()
        mean_pred = y_pred.mean()

        sum_ += 2 * cor * std_true * std_pred / (
            std_true**2 + std_pred**2 + (mean_true - mean_pred)**2
        )
    return sum_ / n

def get_pearson_r_scorer(index):
    if index is None:
        return lambda yt, yp: np.mean([pearsonr(yt[:, j], yp[:, j])[0] for j in range(yt.shape[1])])  #pearsonr(yt, yp)[0].mean()
    else:
        return lambda yt, yp: pearsonr(yt[:, index], yp[:, index])[0]

def get_ccc_scorer(index):
    if index is None:
        return lambda yt, yp: multi_concordance_correlation(yt, yp)
    else:
        return lambda yt, yp: concordance_correlation(yt[:, index], yp[:, index])

def get_scorer(fun, index=None, greater_is_better=True):
    if index is None:
        return lambda yt, yp: fun(yt, yp)
    else:
        return lambda yt, yp: fun(yt[:, index], yp[:, index])
