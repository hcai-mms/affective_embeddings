import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from sklearn import metrics

from services import CACHED_PATH
from services.go_emotion_service import EKMAN_MAPPING
from appendix.latex_service import *
from services.service import Service
from utils.dataset_utils import plutchik_mapping, vad_mapping
from utils.metrics import concordance_correlation_numba

regression_metrics = {
    "ccc": (concordance_correlation_numba, True),
    "r2": (metrics.r2_score, True),
    "mse": (metrics.mean_squared_error, False),
}

classification_metrics = {
    "precision": (metrics.precision_score, True),
    "recall": (metrics.recall_score, True),
    "f1": (metrics.f1_score, True)
}


def regression(dataset, split, model, metric_name, result_path="results"):
    rng = np.random.default_rng(42)
    columns = list(vad_mapping) if dataset == 'vad' else list(plutchik_mapping)
    metric, reverse = regression_metrics[metric_name]

    order = []
    for embedding_name, embedding_path in EMBEDDINGS_REGRESSION.items():
        path = os.path.join(result_path, f"{split}_{dataset}_{embedding_path}_{model}_pred.parquet")
        df = pd.read_parquet(path)
        y_true = df[[f"true_{c}" for c in columns]].to_numpy()
        y_pred = df[[f"pred_{c}" for c in columns]].to_numpy()

        if metric_name == "ccc":
            y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
            y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        order.append((metric(y_true, y_pred), embedding_name))

    order.sort(reverse=reverse)

    embedding_path = EMBEDDINGS_REGRESSION[order[0][1]]
    path = os.path.join(result_path, f"{split}_{dataset}_{embedding_path}_{model}_pred.parquet")
    df = pd.read_parquet(path)
    y_true = df[[f"true_{c}" for c in columns]].to_numpy()

    pred_columns = [f"pred_{c}" for c in columns]
    y_best = df[pred_columns].to_numpy()

    p_values = [-1.0]
    for _, embedding in order[1:]:
        embedding_path = EMBEDDINGS_REGRESSION[embedding]
        path = os.path.join(result_path, f"{split}_{dataset}_{embedding_path}_{model}_pred.parquet")
        df = pd.read_parquet(path)
        y_pred = df[pred_columns].to_numpy()

        diffs = []
        if metric_name == "ccc":
            for b in range(10000):
                idx = rng.choice(len(y_true), size=len(y_true), replace=True)
                yt = np.asarray(y_true[idx], dtype=np.float64).reshape(-1)
                yp = np.asarray(y_pred[idx], dtype=np.float64).reshape(-1)
                yb = np.asarray(y_best[idx], dtype=np.float64).reshape(-1)

                metrica_A = metric(yt, yb)
                metrica_B = metric(yt, yp)
                diffs.append(metrica_A - metrica_B)
        else:
            for b in range(10000):
                idx = rng.choice(len(y_true), size=len(y_true), replace=True)
                metrica_A = metric(y_true[idx], y_best[idx])
                metrica_B = metric(y_true[idx], y_pred[idx])
                diffs.append(metrica_A - metrica_B)
        if reverse:
            p_value = np.mean(np.array(diffs) <= 0)
        else:
            p_value = np.mean(np.array(diffs) >= 0)
        p_values.append(p_value)

    scores = {k:(v,p) for (v,k), p in zip(order, p_values)}
    return split, model, metric_name, scores, order


def paired_permutation_test_swap(
        y_true,
        y_pred_a,
        y_pred_b,
        metric_fn,
        average,
        n_perm=10000,
        seed=42):
    rng = np.random.default_rng(seed)

    score = lambda y_pred: metric_fn(y_true, y_pred, average=average, zero_division=0)

    y_true = np.asarray(y_true)

    metrica_A = score(y_pred_a)
    metrica_B = score(y_pred_b)
    observed = metrica_A - metrica_B

    diffs = np.empty(n_perm, dtype=np.float64)
    n = y_true.shape[0]

    for i in range(n_perm):
        swap = rng.random(n) < 0.5
        a_p = y_pred_a.copy()
        b_p = y_pred_b.copy()
        a_p[swap], b_p[swap] = y_pred_b[swap], y_pred_a[swap]
        diffs[i] = score(a_p) - score(b_p)

    p_one_sided = np.mean(diffs >= observed)
    return observed, p_one_sided


def classification(split, model, metric_name, result_path="results"):
    columns = list(EKMAN_MAPPING)
    metric, reverse = classification_metrics[metric_name]

    order = []
    for embedding_name, embedding_path in EMBEDDINGS_CLASSIFICATION.items():
        path = os.path.join(result_path, f"go_emotion_macro_{embedding_path}_{model}_pred.parquet")
        df = pd.read_parquet(path)
        y_true = df[[f"true_{c}" for c in columns]].to_numpy()
        y_pred = df[[f"pred_{c}" for c in columns]].to_numpy()
        order.append((metric(y_true, y_pred, average=split), embedding_name))

    order.sort(reverse=reverse)

    embedding_path = EMBEDDINGS_CLASSIFICATION[order[0][1]]
    path = os.path.join(result_path, f"go_emotion_macro_{embedding_path}_{model}_pred.parquet")
    df = pd.read_parquet(path)
    y_true = df[[f"true_{c}" for c in columns]].to_numpy()

    pred_columns = [f"pred_{c}" for c in columns]
    y_best = df[pred_columns].to_numpy()

    p_values = [-1.0]
    for _, embedding in order[1:]:
        embedding_path = EMBEDDINGS_CLASSIFICATION[embedding]
        path = os.path.join(result_path, f"go_emotion_macro_{embedding_path}_{model}_pred.parquet")
        df = pd.read_parquet(path)
        y_pred = df[pred_columns].to_numpy()

        observed, p_value = paired_permutation_test_swap(y_true, y_best, y_pred, metric, split)
        p_values.append(p_value)

    scores = {k:(v,p) for (v,k), p in zip(order, p_values)}
    return split, model, metric_name, scores, order


class SignificanceService(Service):
    def __init__(self, dataset, cached_path=CACHED_PATH):
        filepath = os.path.join(cached_path, f'significance_service_{dataset}.pkl')
        self.dataset = dataset
        self.mapping = {}
        super().__init__(filepath)

    def load_regression(self):
        tasks = []
        for split in REGRESSION_SPLITS:
            for model in MODELS:
                for metric in regression_metrics:
                    tasks.append([self.dataset, split, model, metric])

        with ProcessPoolExecutor(max_workers=48) as ex:
            futures = [ex.submit(regression, *t) for t in tasks]
            for f in as_completed(futures):
                result = f.result()
                self.mapping[result[:3]] = result[3:]

    def load_classification(self):
        tasks = []
        for split in ["macro", "weighted", "micro"]:
            for model in MODELS:
                for metric in classification_metrics:
                    tasks.append([split, model, metric])

        with ProcessPoolExecutor(max_workers=48) as ex:
            futures = [ex.submit(classification, *t) for t in tasks]
            for f in as_completed(futures):
                result = f.result()
                self.mapping[result[:3]] = result[3:]

    def load_from_data(self):
        if self.dataset == "go_emotion":
            self.load_classification()
        else:
            self.load_regression()

    def save(self):
        print(f"Saving {len(self.mapping)}", file=sys.stderr)
        self._save(self.mapping)
        print("Done!", file=sys.stderr)

    def load_from_cache(self):
        self.mapping = self._load_from_cache()


def save(dataset):
    significance_service = SignificanceService(dataset)
    significance_service.load_from_data()
    significance_service.save()

def load(dataset):
    significance_service = SignificanceService(dataset)
    significance_service.load_from_cache()
    print("Finished")


if __name__ == '__main__':
    # save("vad")
    # save("vad")
    save("go_emotion")
