import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from sklearn import metrics
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, PredefinedSplit, cross_validate
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit
from torch import nn
from xgboost import XGBRegressor

from utils.pytorch_mlp import PyTorchMLP, HIDDEN_LAYERS
from utils.dataset_utils import get_emotional_columns
from utils.metrics import get_pearson_r_scorer, get_scorer, get_ccc_scorer
from utils.model_utils import extract_dataset, extract_split_method, extract_embedding_name


class RegressionBenchmarkService:
    def __init__(self,
                 splits_parquet_path: str,
                 embedding_path: str,
                 model_name: str,
                 cached_dir='cached',
                 results_dir='results',
                 n_trials=50,
                 model_n_jobs=1,
                 cv_n_jobs=1,
                 optuna_n_jobs=1,
                 seed=42,
                 timeout=90 * 60):

        self.splits_parquet_path = splits_parquet_path
        self.embedding_path = embedding_path
        self.model_name = model_name
        self.cached_dir = cached_dir
        self.results_dir = results_dir
        self.n_trials = n_trials
        self.model_n_jobs = model_n_jobs
        self.cv_n_jobs = cv_n_jobs
        self.optuna_n_jobs = optuna_n_jobs
        self.splits = {}
        self.embedding_dim = 0
        self.seed = seed
        self.timeout = timeout

        self.logger = logging.getLogger("RegressionBenchmarkService")

        self.dataset = extract_dataset(splits_parquet_path)
        self.split_method = extract_split_method(splits_parquet_path)
        self.emotion_columns = get_emotional_columns(self.dataset)
        self.embedding_name = extract_embedding_name(embedding_path)

        self.experiment_name = f"{self.split_method}_{self.dataset}_{self.embedding_name}_{self.model_name}"
        sql_storage_dir = os.path.join(self.cached_dir, "optuna")
        self.sql_storage_path = Path(os.path.join(sql_storage_dir, f"{self.experiment_name}.db")).resolve()
        self.sql_storage_url = f"sqlite:///{self.sql_storage_path}"

        os.makedirs(sql_storage_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        self.logger.info(f"=== Regression Benchmark ===")
        self.logger.info(f"\tDataset: {self.dataset.upper()}")
        self.logger.info(f"\tExperiment: {self.experiment_name}")
        self.logger.info(f"\tSplit method: {self.split_method}")
        self.logger.info(f"\tEmbedding: {self.embedding_name}")
        self.logger.info(f"\tModel: {self.model_name}")

    def get_elasticnet_model(self, alpha, l1_ratio):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('elasticnet', MultiTaskElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=10000,
                tol=1e-3,
                selection='random',
                random_state=42
            ))
        ])

    def get_knn_model(self, n_neighbors, weights):
        return KNeighborsRegressor(
                n_neighbors=n_neighbors,
                weights=weights,
                metric="cosine",
                algorithm='brute',
                n_jobs=self.model_n_jobs
            )

    def get_xgboost_model(self, learning_rate, max_depth, reg_lambda):
        base_estimator = XGBRegressor(
            tree_method='hist',
            device='cuda',
            n_estimators=200,
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_bin=128,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=reg_lambda,
            random_state=42,
            verbosity=0
        )
        return MultiOutputRegressor(base_estimator)

    def get_mlp_model(self, complexity, activation, lr, weight_decay):
        if self.dataset == 'plutchik':
            batch_size = 256
            final_activation = 'logistic'
        else:
            batch_size = 4096
            final_activation = 'tanh'

        return Pipeline([
            ('scaler', MinMaxScaler(feature_range=(-1, 1))),
            ('mlp', NeuralNetRegressor(
                module=PyTorchMLP,
                module__input_dim=self.embedding_dim,
                module__output_dim=len(self.emotion_columns),
                module__complexity=complexity,
                module__activation=activation,
                module__final_activation=final_activation,
                criterion=nn.MSELoss,
                module__dropout=0,
                lr=lr,
                optimizer=torch.optim.Adam,
                optimizer__weight_decay=weight_decay,
                max_epochs=2000,
                batch_size=batch_size,
                iterator_train__shuffle=True,
                train_split=ValidSplit(cv=0.1, random_state=42),
                verbose=0,
                device='cuda',
                callbacks=[
                    ('early_stopping', EarlyStopping(patience=8, monitor='valid_loss'))
                ]
            ))
        ])

    def get_model_for_optimization(self, trial: optuna.Trial):
        match self.model_name:
            case 'elasticnet':
                alpha = trial.suggest_float('alpha', 1e-4, 10.0, log=True)
                l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
                return self.get_elasticnet_model(alpha, l1_ratio)

            case 'knn':
                n_neighbors = trial.suggest_int('n_neighbors', 1, 100)
                weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
                return self.get_knn_model(n_neighbors, weights)

            case 'xgboost':
                learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
                max_depth = trial.suggest_int('max_depth', 3, 10)
                reg_lambda = trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True)
                return self.get_xgboost_model(learning_rate, max_depth, reg_lambda)

            case 'mlp':
                complexity = trial.suggest_categorical('complexity', range(len(HIDDEN_LAYERS)))
                activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic'])
                lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
                weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
                return self.get_mlp_model(complexity, activation, lr, weight_decay)

            case _:
                raise ValueError(f"Unknown model name: {self.model_name}")

    def get_model_for_evaluation(self, params: dict):
        match self.model_name:
            case 'elasticnet':
                return self.get_elasticnet_model(**params)
            case 'knn':
                return self.get_knn_model(**params)
            case 'xgboost':
                return self.get_xgboost_model(**params)
            case 'mlp':
                return self.get_mlp_model(**params)
            case _:
                raise ValueError(f"Unknown model name: {self.model_name}")

    def load_data(self):
        if self.splits:
            self.logger.warning(f"Data already loaded -> skipping")
            return

        self.logger.info(f"Loading embeddings from {self.embedding_path}...")
        df_embeddings = pd.read_parquet(self.embedding_path)
        embeddings = {
            word: np.array(row.values) / np.linalg.norm(row.values)
            for word, row in df_embeddings.iterrows()
        }

        self.logger.info(
            f"\tLoaded {len(embeddings)} embeddings with dimension {next(iter(embeddings.values())).shape[0]}")

        self.logger.info(f"Loading training data from {self.splits_parquet_path}...")
        df = pd.read_parquet(self.splits_parquet_path)
        n_total = len(df)
        self.logger.info(f"\tLoaded {len(embeddings)} training with {n_total} rows")
        self.logger.info(f"\tColumns {len(list(df.columns))}")

        assert len(df['word'].unique()) == len(df)
        n_excluded = len(df[df['fold_idx'] == -2])

        self.logger.info(f"\tExcluding {n_excluded} items not in WordNet")
        train_df = df[df['fold_idx'] > -1].reset_index(drop=True)
        test_df = df[df['fold_idx'] == -1].reset_index(drop=True)

        X_train = df_embeddings.loc[train_df["word"]].to_numpy(dtype=np.float32)
        y_train = train_df[self.emotion_columns].to_numpy(dtype=np.float32)
        ps = PredefinedSplit(test_fold=train_df["fold_idx"].to_numpy())

        X_test = df_embeddings.loc[test_df["word"]].to_numpy(dtype=np.float32)
        y_test = test_df[self.emotion_columns].to_numpy(dtype=np.float32)

        assert len(df) == len(X_train) + len(X_test) + n_excluded
        self.embedding_dim = X_train.shape[1]

        self.splits = {
            'X_train': X_train,
            'y_train': y_train,
            'cv': ps,
            'X_test': X_test,
            'y_test': y_test
        }

    def run_training(self):
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        if os.path.exists(self.sql_storage_path):
            study = optuna.load_study(
                storage=self.sql_storage_url,
                study_name=self.experiment_name
            )
            completed = sum(t.state == TrialState.COMPLETE for t in study.trials)
            n_remaining_trails = self.n_trials - completed
            self.logger.info(f"\tAlready trained {completed}/{self.n_trials}")

        else:
            study = optuna.create_study(
                storage=self.sql_storage_url,
                study_name=self.experiment_name,
                sampler=TPESampler(seed=self.seed),
                direction='maximize',
                load_if_exists=False
            )
            n_remaining_trails = self.n_trials
        if n_remaining_trails <= 0:
            self.logger.info(f"=== COMPLETED: Nothing more to do! ===")
            return

        self.load_data()

        def objective(trial):
            model = self.get_model_for_optimization(trial)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="This Pipeline instance is not fitted yet",
                    category=FutureWarning,
                )
                scores = cross_val_score(
                    model,
                    self.splits['X_train'],
                    self.splits['y_train'],
                    cv=self.splits['cv'],
                    scoring="r2",
                    n_jobs=self.cv_n_jobs)
                return scores.mean()

        study.optimize(
            objective,
            n_trials=n_remaining_trails,
            show_progress_bar=True,
            n_jobs=self.optuna_n_jobs,
            timeout=self.timeout
        )

        self.logger.info(f"\tBest params: {study.best_params}")
        self.logger.info(f"\tBest validation score: {study.best_value:.6f} (R2)")

    def get_study(self) -> optuna.Study:
        return optuna.load_study(
            storage=self.sql_storage_url,
            study_name=self.experiment_name
        )

    def run_evaluation(self):
        study = self.get_study()
        completed = sum(t.state == TrialState.COMPLETE for t in study.trials)
        if self.n_trials > completed:
            raise ValueError(f"n_trails = {self.n_trials} and study trails = {completed} do not match!")

        self.load_data()

        best_params = study.best_params
        best_model = self.get_model_for_evaluation(best_params)

        scoring = {}
        for i, emotion in enumerate(["avg"] + self.emotion_columns):
            if i == 0:
                index = None
            else:
                index = i - 1
            scoring[f"mae_{emotion}"] = get_scorer(metrics.mean_absolute_error, index)
            scoring[f"mse_{emotion}"] = get_scorer(metrics.mean_squared_error, index)
            scoring[f"rmse_{emotion}"] = get_scorer(metrics.root_mean_squared_error, index)
            scoring[f"r2_{emotion}"] = get_scorer(metrics.r2_score, index)
            scoring[f"pearsonr_{emotion}"] = get_pearson_r_scorer(index)
            scoring[f"ccc_{emotion}"] = get_ccc_scorer(index)
        validation_scorer = { k: make_scorer(f, greater_is_better=k.startswith(("r2_", "pearson_r", "ccc_")))
                              for k, f in scoring.items()}

        result = cross_validate(
            best_model,
            self.splits["X_train"],
            self.splits["y_train"],
            cv=self.splits["cv"],
            scoring=validation_scorer,
            return_estimator=True,
            return_train_score=False,
            n_jobs=self.cv_n_jobs
        )

        data_labels = sorted(list(scoring))
        columns = [f"fold_{i}" for i in range(5)]
        validation_data = [result["test_" + label] for label in data_labels]
        data_labels.append("n_iterations")
        validation_data.append([self.get_iteration_count(estimator) for estimator in result["estimator"]])
        validation_df = pd.DataFrame(validation_data, index=data_labels, columns=columns)
        validation_path = os.path.join(self.results_dir, f"{self.experiment_name}_train.parquet")
        self.logger.info(f"Saving validation report to: {validation_path}...")
        validation_df.to_parquet(validation_path)

        best_model.fit(self.splits["X_train"], self.splits["y_train"])
        y_pred = best_model.predict(self.splits["X_test"])
        y_true = self.splits["y_test"]

        pred_columns = [f"pred_{c}" for c in self.emotion_columns] + [f"true_{c}" for c in self.emotion_columns]
        pred_df = pd.DataFrame(np.concat([y_pred, y_true], axis=1), columns=pred_columns)
        pred_path = os.path.join(self.results_dir, f"{self.experiment_name}_pred.parquet")
        pred_df.to_parquet(pred_path)

        test_data = {k: f(y_true, y_pred) for k, f in scoring.items()}
        index = [k.split("_")[0] for k in test_data if k.endswith("_avg")]
        columns = self.emotion_columns + ["avg"]
        rows = [[test_data[f"{k}_{emotion}"] for emotion in columns] for k in index]

        test_df = pd.DataFrame(rows, index=index, columns=columns)
        test_path = os.path.join(self.results_dir, f"{self.experiment_name}_test.parquet")
        self.logger.info(f"Saving test report to: {test_path}...")
        test_df.to_parquet(test_path)

    def get_iteration_count(self, model):
        if self.model_name == 'elasticnet':
            # MultiTaskElasticNet stores iteration count in n_iter_
            try:
                return int(model.named_steps['elasticnet'].n_iter_)
            except (AttributeError, KeyError):
                return None

        elif self.model_name == 'mlp':
            # skorch NeuralNetRegressor stores training history
            try:
                # Get the last epoch number from history
                history = model.named_steps['mlp'].history
                if len(history) > 0:
                    return int(history[-1, 'epoch'])
                return None
            except (AttributeError, KeyError, IndexError):
                return None
        else:
            # XGBoost uses fixed n_estimators, KNN has no iterations (no iteration tracking)
            return None


def parse_args_and_run():
    parser = argparse.ArgumentParser(
        description='Run regression benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Training elasticnet on plutchik with morphological blocking
  python -m services.regression_benchmark_service (for training elasticnet on plutchik)\\
      --embedding-path embeddings/embeddings_openai@text-embedding-3-large_api.parquet \\
      --splits-path opendata/splits/cluster_splits_5fold_morphological_vad.parquet \\
      --model elasticnet \\
      --n-trials 100 \\
      --train \\
      --n-jobs 1 \\
      --optuna-jobs 48

Examples:
  Testing xboost on vad with semantic blocking
  python -m services.regression_benchmark_service \\
      --embedding-path embeddings/embeddings_sentence-transformers@sentence-t5-xxl.parquet \\
      --splits-path opendata/splits/cluster_splits_5fold_community_vad.parquet \\
      --model xgboost \\
      --n-trials 100 \\
      --test \\
      --n-jobs 1 \\
      --optuna-jobs 48

Output files:
  - Hyperparameter caching: cached/optuna/community_plutchik_openai@text-embedding-3-large_elasticnet.db
  - Validation report: results/community_plutchik_openai@text-embedding-3-large_elasticnet_train.parquet
  - Test report: results/community_plutchik_sentence-transformers_sentence-t5-xxl_xgboost_test.parquet
        """
    )

    parser.add_argument(
        '--embedding-path',
        type=str,
        required=True,
        help='Path to embedding parquet file'
    )

    parser.add_argument(
        '--splits-path',
        type=str,
        required=True,
        help='Path to splits parquet file (dataset auto-detected from filename: vad/plutchik)'
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['elasticnet', 'knn', 'xgboost', 'mlp'],
        help='Model to benchmark (xgboost and mlp use GPU acceleration)'
    )

    parser.add_argument(
        '--n-trials',
        type=int,
        default=100,
        help='Number of Optuna trials for hyperparameter optimization (default: 100)'
    )

    parser.add_argument(
        '--model-jobs',
        type=int,
        default=1,
        help='Number of CPU cores for model training (default: 1 = sequential)'
    )

    parser.add_argument(
        '--cv-jobs',
        type=int,
        default=1,
        help='Number of parallel jobs for cross-validation (default: 1 = sequential, recommended: 5)'
    )

    parser.add_argument(
        '--optuna-jobs',
        type=int,
        default=1,
        help='Number of parallel Optuna trials (default: 1 = sequential)'
    )

    parser.add_argument(
        '--timeout',
        type=int,
        default=90,
        help='Timeout in minutes after which the training automatically stops'
    )

    parser.add_argument(
        '--cached-dir',
        type=str,
        default='cached',
        help='Directory for pickle files (default: cached)'
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory for parquet reports (default: results)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    parser.add_argument(
        '--train',
        action='store_true',
        help='Train the model')

    parser.add_argument(
        '--test',
        action='store_true',
        help='Train the model')

    args = parser.parse_args()

    if not args.train and not args.test:
        print(f"ERROR: Define --train or --test", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.embedding_path):
        print(f"ERROR: Embedding file not found: {args.embedding_path}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.splits_path):
        print(f"ERROR: Splits file not found: {args.splits_path}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.embedding_path):
        print(f"ERROR: Embedding file not found: {args.splits_path}", file=sys.stderr)
        sys.exit(1)

    # Create and run benchmark
    regressor = RegressionBenchmarkService(
        splits_parquet_path=args.splits_path,
        embedding_path=args.embedding_path,
        model_name=args.model,
        cached_dir=args.cached_dir,
        results_dir=args.results_dir,
        n_trials=args.n_trials,
        model_n_jobs=args.model_jobs,
        cv_n_jobs=args.cv_jobs,
        optuna_n_jobs=args.optuna_jobs,
        timeout=60*args.timeout
    )

    if args.train:
        regressor.run_training()
    if args.test:
        regressor.run_evaluation()


def main():
    logging.basicConfig(level=logging.INFO)
    start_time = time.time()
    try:
        parse_args_and_run()
    except Exception as ex:
        print(">>> Program failed after:", time.time() - start_time)
        raise ex
    print(">>> Program completed after:", time.time() - start_time)


if __name__ == '__main__':
    main()
