from pathlib import Path

import numpy as np
import os
import pandas as pd
import optuna
from optuna.trial import TrialState

from services.latex_service import *

EMBEDDINGS_REGRESSION = {
    "ST5 XLL": "sentence-transformers_sentence-t5-xxl",
    "EmbeddingGemma": "google_embeddinggemma-300m(STS",
    "Nomic v2": "nomic-ai_nomic-embed-text-v2-moe",
    "Multilingual E5 L Instr": "intfloat_multilingual-e5-large-instruct(retrieval)",
    "Jina v4": "jinaai_jina-embeddings-v4",
    "Ling Mistral": "Linq-AI-Research_Linq-Embed-Mistral(retrieval)",
    "KaLM v2": "tencent_KaLM-Embedding-Gemma3-12B-2511(retrieval)",
    "LLaMA Nemotron 8B": "nvidia_llama-embed-nemotron-8b",
    "Qwen3 8B": "Qwen_Qwen3-Embedding-8B(retrieval)",
    "Voyage v3 L": "voyage_voyage-3.5",
    "OpenAI Text v3 L": "openai_text-embedding-3-large",
    "Gemini 001": "google_gemini-embedding-001",
}

EMBEDDINGS_CLASSIFICATION = {
    "ST5 XLL": "sentence-transformers_sentence-t5-xxl",
    "EmbeddingGemma": "google_embeddinggemma-300m",
    "Nomic v2": "nomic-ai_nomic-embed-text-v2-moe",
    "Multilingual E5 L Instr": "intfloat_multilingual-e5-large-instruct",
    "Jina v4": "jinaai_jina-embeddings-v4",
    "Ling Mistral": "Linq-AI-Research_Linq-Embed-Mistral",
    "KaLM v2": "tencent_KaLM-Embedding-Gemma3-12B-2511",
    "LLaMA Nemotron 8B": "nvidia_llama-embed-nemotron-8b",
    "Qwen3 8B": "Qwen_Qwen3-Embedding-8B",
    "Voyage v3 L": "voyage_voyage-3.5",
    "OpenAI Text v3 L": "openai_text-embedding-3-large",
    "Gemini 001": "google_gemini-embedding-001",
}

def get_trails(sql_storage_path, experiment_name):
    if os.path.exists(sql_storage_path):
        sql_storage_url = f"sqlite:///{sql_storage_path}"
        study = optuna.load_study(
            storage=sql_storage_url,
            study_name=experiment_name
        )
        return sum(t.state == TrialState.COMPLETE for t in study.trials), study.best_value, study.best_params
    else:
        return None

def fmt(x):
    if pd.isna(x):
        return ""
    if isinstance(x, (int, float, np.floating, np.integer)):
        ax = abs(float(x))
        if 0 < ax < 1e-2:
            return f"{x:.1e}"     # exponential below 0.001 (but not 0)
        return f"{x:.3f}"         # 3 decimals otherwise
    return str(x)


def generate_table_regression(dataset, split, model, sql_storage_dir="cached/optuna"):
    rows = []
    columns = []
    for embedding_name, embedding in EMBEDDINGS_REGRESSION.items():
        experiment_name = f"{split}_{dataset}_{embedding}_{model}"
        sql_storage_path = Path(os.path.join(sql_storage_dir, f"{experiment_name}.db")).resolve()
        result = get_trails(sql_storage_path, experiment_name)
        trails, best_value, best_param = result
        if not columns:
            columns = [k.replace('_', '-') for k in best_param]
        rows.append([best_value] + list(best_param.values()))
    columns = ["$R^2$"] + columns
    df = pd.DataFrame(rows, columns=columns, index=list(EMBEDDINGS_REGRESSION))

    if model == "mlp":
        print("\\begin{table*}[!htb]")
        print("\\centering")
    else:
        print("\\begin{table}[!htb]")
        print("\\resizebox{\\linewidth}{!}{")
    print(df.to_latex(float_format=fmt), end='')
    split_name = "semantic" if split == "community" else "morphological"
    dataset_name = "NRC-VAD" if dataset == "vad" else "NRC-EIL"
    match model:
        case "mlp": model_name = "MLP"
        case "knn": model_name = "$k$-NN"
        case "xgboost": model_name = "XGBoost"
        case _: model_name = "ElasticNet"

    caption = "\\caption{" + f"Summary of the best hyperparameters for {model_name} on the {split_name} splitting strategy for {dataset_name} dataset." + "}"

    if model == "mlp":
        print(caption)
        print("\\end{table*}")
    else:
        print("}")
        print(caption)
        print("\\end{table}")

def generate_table_classification(model, sql_storage_dir="cached/optuna"):
    rows = []
    columns = []
    for embedding_name, embedding in EMBEDDINGS_CLASSIFICATION.items():
        experiment_name = f"go_emotion_macro_{embedding}_{model}"
        sql_storage_path = Path(os.path.join(sql_storage_dir, f"{experiment_name}.db")).resolve()
        result = get_trails(sql_storage_path, experiment_name)
        trails, best_value, best_param = result
        if not columns:
            columns = [k.replace('_', '-') for k in best_param]
        rows.append([best_value] + list(best_param.values()))
    columns = ["$F1$"] + columns
    df = pd.DataFrame(rows, columns=columns, index=list(EMBEDDINGS_CLASSIFICATION))

    if model == "mlp":
        print("\\begin{table*}[!htb]")
        print("\\centering")
    else:
        print("\\begin{table}[!htb]")
        print("\\resizebox{\\linewidth}{!}{")
    print(df.to_latex(float_format=fmt), end='')
    match model:
        case "mlp": model_name = "MLP"
        case "knn": model_name = "$k$-NN"
        case "xgboost": model_name = "XGBoost"
        case _: model_name = "logistic regression with elasticnet regularization"
    caption = "\\caption{" + f"Summary of the best hyperparameters for {model_name} on the GoEmotions dataset." + "}"

    if model == "mlp":
        print(caption)
        print("\\end{table*}")
    else:
        print("}")
        print(caption)
        print("\\end{table}")


def main():
    for model in MODELS:
        for dataset in REGRESSION_DATASETS:
            for split in REGRESSION_SPLITS:
                print()
                generate_table_regression(dataset, split, model)
                print()
        generate_table_classification(model)


if __name__ == '__main__':
    main()
