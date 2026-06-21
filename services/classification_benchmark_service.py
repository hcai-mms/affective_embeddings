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
from sklearn import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import PredefinedSplit, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MultiLabelBinarizer
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit
from xgboost import XGBClassifier

from services.go_emotion_service import EKMAN_MAPPING, EMOTIONS
from utils.model_utils import extract_embedding_name
from utils.pytorch_mlp import PyTorchMLP, HIDDEN_LAYERS, NeuralNetBCE


class ClassificationBenchmarkService:
    def __init__(self,
                 splits_parquet_path: str,
                 category: str,
                 embedding_path: str,
                 model_name: str,
                 cached_dir='cached',
                 results_dir='results',
                 n_trials=50,
                 model_n_jobs=1,
                 cv_n_jobs=1,
                 optuna_n_jobs=1,
                 seed=42,
                 timeout=90 * 60,
                 threshold=0.5):

        self.splits_parquet_path = splits_parquet_path
        self.category = category
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
        self.threshold = threshold

        self.logger = logging.getLogger("ClassificationBenchmarkService")

        self.dataset = "go_emotion"
        if self.category == "micro":
            self.emotion_columns = list(EMOTIONS)
        elif self.category == "macro":
            self.emotion_columns = list(EKMAN_MAPPING)
        else:
            raise ValueError(f"Category {self.category} is neither major nor minor")
        self.embedding_name = extract_embedding_name(embedding_path)

        self.experiment_name = f"{self.dataset}_{self.category}_{self.embedding_name}_{self.model_name}"
        sql_storage_dir = os.path.join(self.cached_dir, "optuna")
        self.sql_storage_path = Path(os.path.join(sql_storage_dir, f"{self.experiment_name}.db")).resolve()
        self.sql_storage_url = f"sqlite:///{self.sql_storage_path}"

        os.makedirs(sql_storage_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        self.logger.info(f"=== Classification Benchmark ===")
        self.logger.info(f"\tDataset: {self.dataset.upper()}")
        self.logger.info(f"\tCategory: {self.category}")
        self.logger.info(f"\tExperiment: {self.experiment_name}")
        self.logger.info(f"\tEmbedding: {self.embedding_name}")
        self.logger.info(f"\tModel: {self.model_name}")

    def get_elasticnet_model(self, C, l1_ratio):
        base = LogisticRegression(
            C=C,
            l1_ratio=l1_ratio,
            penalty="elasticnet",
            solver="saga",
            max_iter=10000,
            tol=1e-3,
            random_state=42,
        )
        return Pipeline([
            ('scaler', StandardScaler()),
            ('ovr', OneVsRestClassifier(base, n_jobs=self.model_n_jobs)),
        ])

    def get_knn_model(self, n_neighbors, weights):
        return KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric="cosine",
            algorithm='brute',
            n_jobs=self.model_n_jobs
        )

    def get_xgboost_model(self, learning_rate, max_depth, reg_lambda):
        base = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method='hist',
            device='cuda',
            n_estimators=100,
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
        return OneVsRestClassifier(base, n_jobs=self.model_n_jobs)

    def get_mlp_model(self, complexity, activation, lr, weight_decay):
        return Pipeline([
            ('scaler', MinMaxScaler(feature_range=(-1, 1))),
            ('mlp', NeuralNetBCE(
                module=PyTorchMLP,
                module__input_dim=self.embedding_dim,
                module__output_dim=len(self.emotion_columns),
                module__complexity=complexity,
                module__activation=activation,
                criterion=torch.nn.BCEWithLogitsLoss,
                module__dropout=0,
                lr=lr,
                optimizer=torch.optim.Adam,
                optimizer__weight_decay=weight_decay,
                max_epochs=5000,
                batch_size=4096,
                iterator_train__shuffle=True,
                train_split=ValidSplit(cv=0.1, random_state=42),
                verbose=0,
                device='cuda',
                callbacks=[
                    ('early_stopping', EarlyStopping(patience=16, monitor='valid_loss'))
                ]
            ))
        ])

    def get_model_for_optimization(self, trial: optuna.Trial):
        match self.model_name:
            case 'elasticnet':
                C = trial.suggest_float('C', 1e-4, 100.0, log=True)
                l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
                return self.get_elasticnet_model(C, l1_ratio)

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

        if self.category == "micro":
            category = df["category"]
        else:
            index_mapping = {EMOTIONS.index(e): self.emotion_columns.index(k) for k, v in EKMAN_MAPPING.items() for e in
                             v}
            category = df["category"].apply(lambda x: [index_mapping[e] for e in x])

        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(category)
        y_df = pd.DataFrame(y, index=df.index)

        train_df = df[df['fold_idx'] > -1]
        test_df = df[df['fold_idx'] == -1]

        X_train = df_embeddings.loc[train_df["sentence_text"]].to_numpy(dtype=np.float32)
        X_test = df_embeddings.loc[test_df["sentence_text"]].to_numpy(dtype=np.float32)

        y_train = y_df.loc[train_df.index].to_numpy(dtype=np.float32)
        y_test = y_df.loc[test_df.index].to_numpy(dtype=np.float32)
        ps = PredefinedSplit(test_fold=train_df["fold_idx"].to_numpy())

        self.embedding_dim = X_train.shape[1]
        self.splits = {
            'X_train': X_train,
            'y_train': y_train,
            'cv': ps,
            'X_test': X_test,
            'y_test': y_test
        }

    def get_y_pred(self, y_score):
        if self.model_name == "knn":
            y_score = np.column_stack([p[:, 1] for p in y_score])
        return (y_score >= self.threshold).astype(np.int8)

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
                y_score = cross_val_predict(
                    model,
                    self.splits['X_train'],
                    self.splits['y_train'],
                    cv=self.splits['cv'],
                    method="predict_proba",
                    n_jobs=self.cv_n_jobs)

                y_pred = self.get_y_pred(y_score)
                return f1_score(self.splits['y_train'], y_pred, average="macro", zero_division=0)

        study.optimize(
            objective,
            n_trials=n_remaining_trails,
            show_progress_bar=True,
            n_jobs=self.optuna_n_jobs,
            timeout=self.timeout
        )

        self.logger.info(f"\tBest params: {study.best_params}")
        self.logger.info(f"\tBest validation score: {study.best_value:.6f} (LogLoss)")

    def get_study(self) -> optuna.Study:
        return optuna.load_study(
            storage=self.sql_storage_url,
            study_name=self.experiment_name
        )

    def cv_per_fold_reports(self, model, X, y, cv):
        fold_reports = []
        for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
            m = clone(model)
            m.fit(X[tr], y[tr])

            y_score = m.predict_proba(X[va])
            y_pred = self.get_y_pred(y_score)

            rep = classification_report(
                y[va], y_pred,
                target_names=self.emotion_columns,
                zero_division=0,
                output_dict=True
            )
            fold_reports.append(rep)
        return fold_reports

    def run_evaluation(self):
        study = self.get_study()
        completed = sum(t.state == TrialState.COMPLETE for t in study.trials)
        if self.n_trials > completed:
            raise ValueError(f"n_trails = {self.n_trials} and study trails = {completed} do not match!")
        self.load_data()

        best_params = study.best_params
        best_model = self.get_model_for_evaluation(best_params)

        fold_report = self.cv_per_fold_reports(best_model, self.splits["X_train"],
                                               self.splits["y_train"], cv=self.splits["cv"])

        y_score = cross_val_predict(
            best_model,
            self.splits["X_train"],
            self.splits["y_train"],
            cv=self.splits["cv"],
            method="predict_proba",
            n_jobs=self.cv_n_jobs,
        )
        y_pred = self.get_y_pred(y_score)

        validation_report = classification_report(
            self.splits["y_train"],
            y_pred,
            target_names=self.emotion_columns,
            zero_division=0,
            output_dict=True
        )

        dict_ = {}
        for emotion, score_dict in validation_report.items():
            for metric, score in score_dict.items():
                dict_[f"{metric}_{emotion}"] = [score]

        for fold in fold_report:
            for emotion, score_dict in fold.items():
                for metric, score in score_dict.items():
                    dict_[f"{metric}_{emotion}"].append(score)

        validation_df = pd.DataFrame.from_dict(dict_, orient="index", columns=["avg"] + [f"fold_{i}" for i in range(5)])
        validation_path = os.path.join(self.results_dir, f"{self.experiment_name}_train.parquet")
        self.logger.info(f"Saving validation report to: {validation_path}...")
        validation_df.to_parquet(validation_path)

        best_model.fit(self.splits["X_train"], self.splits["y_train"])
        y_score = best_model.predict_proba(self.splits["X_test"])
        y_pred = self.get_y_pred(y_score)

        pred_columns = [f"pred_{c}" for c in self.emotion_columns] + [f"true_{c}" for c in self.emotion_columns]
        pred_df = pd.DataFrame(np.concat([y_pred, self.splits["y_test"]], axis=1), columns=pred_columns)
        pred_path = os.path.join(self.results_dir, f"{self.experiment_name}_pred.parquet")
        pred_df.to_parquet(pred_path)

        test_report = classification_report(
            self.splits["y_test"],
            y_pred,
            target_names=self.emotion_columns,
            zero_division=0,
            output_dict=True
        )

        test_df = pd.DataFrame.from_dict(test_report)
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
        description='Run classification benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter)

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
        '--category',
        type=str,
        required=True,
        help="Granularity of emotion (micro -> 27 emotions|macro -> Ekman's taxonomy)"
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

    if args.category not in ["macro", "micro"]:
        print(f"ERROR: Category must be either macro or micro", file=sys.stderr)
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
    regressor = ClassificationBenchmarkService(
        splits_parquet_path=args.splits_path,
        category=args.category,
        embedding_path=args.embedding_path,
        model_name=args.model,
        cached_dir=args.cached_dir,
        results_dir=args.results_dir,
        n_trials=args.n_trials,
        model_n_jobs=args.model_jobs,
        cv_n_jobs=args.cv_jobs,
        optuna_n_jobs=args.optuna_jobs,
        timeout=60 * args.timeout
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
