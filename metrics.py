import numpy as np
from sklearn.metrics import (normalized_mutual_info_score, adjusted_rand_score,
                              fowlkes_mallows_score, accuracy_score)
from scipy.optimize import linear_sum_assignment


def hungarian_acc(y_true, y_pred):
    n = max(y_true.max(), y_pred.max()) + 1
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    row, col = linear_sum_assignment(-cm)
    mapping = {c: r for r, c in zip(row, col)}
    y_remapped = np.array([mapping.get(p, p) for p in y_pred])
    return accuracy_score(y_true, y_remapped)


def calc_metrics(y_true, y_pred):
    nmi = normalized_mutual_info_score(y_true, y_pred) * 100
    ari = adjusted_rand_score(y_true, y_pred) * 100
    acc = hungarian_acc(y_true, y_pred) * 100
    fmi = fowlkes_mallows_score(y_true, y_pred) * 100
    return {'NMI': nmi, 'ARI': ari, 'ACC': acc, 'FMI': fmi,
            'Avg': (nmi + ari + acc + fmi) / 4}


def set_seed(seed):
    import torch, random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
