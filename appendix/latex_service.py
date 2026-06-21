import os

import numpy as np
import pandas as pd

MODELS = ["elasticnet", "knn", "xgboost", "mlp"]
REGRESSION_SPLITS = ["community", "morphological", ]
REGRESSION_DATASETS = ["vad", "plutchik"]

EMBEDDINGS_REGRESSION = {
    "Multilingual e5 L": "intfloat_multilingual-e5-large-instruct(retrieval)",
    "KaLM v2": "tencent_KaLM-Embedding-Gemma3-12B-2511(retrieval)",
    "SRF": "Linq-AI-Research_Linq-Embed-Mistral(retrieval)",
    "Qwen3 8B": "Qwen_Qwen3-Embedding-8B(retrieval)",
    "OpenAI Text v3 L": "openai_text-embedding-3-large",
    "LLaMA Nemotron 8B": "nvidia_llama-embed-nemotron-8b",
    "Gemini 001": "google_gemini-embedding-001",
    "ST5 XLL": "sentence-transformers_sentence-t5-xxl",
    "Voyage v3 L": "voyage_voyage-3.5",
    "EmbeddingGemma": "google_embeddinggemma-300m(STS",
    "Jina v4": "jinaai_jina-embeddings-v4",
    "Nomic v2": "nomic-ai_nomic-embed-text-v2-moe",
}

EMBEDDINGS_CLASSIFICATION = {
    "Multilingual e5 L": "intfloat_multilingual-e5-large-instruct",
    "KaLM v2": "tencent_KaLM-Embedding-Gemma3-12B-2511",
    "SRF": "Linq-AI-Research_Linq-Embed-Mistral",
    "Qwen3 8B": "Qwen_Qwen3-Embedding-8B",
    "OpenAI Text v3 L": "openai_text-embedding-3-large",
    "LLaMA Nemotron 8B": "nvidia_llama-embed-nemotron-8b",
    "Gemini 001": "google_gemini-embedding-001",
    "ST5 XLL": "sentence-transformers_sentence-t5-xxl",
    "Voyage v3 L": "voyage_voyage-3.5",
    "EmbeddingGemma": "google_embeddinggemma-300m",
    "Jina v4": "jinaai_jina-embeddings-v4",
    "Nomic v2": "nomic-ai_nomic-embed-text-v2-moe",
}

VAD_ORDER = [
    "KaLM v2",
    "SRF",
    "OpenAI Text v3 L",
    "Qwen3 8B",
    "LLaMA Nemotron 8B",
    "Gemini 001",
    "ST5 XLL",
    "EmbeddingGemma",
    "Voyage v3 L",
    "Multilingual e5 L",
    "Jina v4",
    "Nomic v2",
]

PLUTCHIK_ORDER = [
    "KaLM v2",
    "SRF",
    "Qwen3 8B",
    "EmbeddingGemma",
    "LLaMA Nemotron 8B",
    "OpenAI Text v3 L",
    "ST5 XLL",
    "Multilingual e5 L",
    "Gemini 001",
    "Voyage v3 L",
    "Jina v4",
    "Nomic v2",
]

GO_EMOTION_ORDER = [
    "Gemini 001",
    "EmbeddingGemma",
    "OpenAI Text v3 L",
    "SRF",
    "Qwen3 8B",
    "KaLM v2",
    "Multilingual e5 L",
    "LLaMA Nemotron 8B",
    "Nomic v2",
    "Jina v4",
    "Voyage v3 L",
    "ST5 XLL",
]

REGRESSION_METRICS = ["mse", "r2", "ccc"]
CLASSIFICATION_METRICS = ["precision", "recall", "f1-score"]

def regression(dataset, result_path="results"):
    row = ["\\multicolumn{3}{c}{" + metric + "}" for metric in ["LR", "$k$-NN", "XGB", "MLP"]]
    print(f'& {" & ".join(row)}\\\\')
    row = []
    for _ in range(4):
        row += ["$MSE$", "$R^2$", "$\\rho_{c}$"]
    print(f'& {" & ".join(row)}\\\\')
    order = VAD_ORDER if dataset == "vad" else PLUTCHIK_ORDER
    for embedding in order:
        embedding_path = EMBEDDINGS_REGRESSION[embedding]
        print(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}")
        for i, split in enumerate(REGRESSION_SPLITS):
            row = []
            for model in MODELS:
                path = os.path.join(result_path, f"{split}_{dataset}_{embedding_path}_{model}_train.parquet")
                df = pd.read_parquet(path)

                for metric in REGRESSION_METRICS:
                    scores = df.loc[metric+"_avg"].tolist()
                    row.append("$" + f"{np.mean(scores):.3f}"[1:] + "\\pm" + f"{np.std(scores):.3f}$"[1:])
            if i == 0:
                embedding_col = r"\multirow{2}{*}{" + embedding + "}"
            else:
                embedding_col = ""
            print(f'{embedding_col} & {" & ".join(row)}\\\\')


def regression1(dataset, result_path="results"):
    row = ["\\multicolumn{3}{c}{" + metric + "}" for metric in ["LR", "$k$-NN"]]
    print(f'& {" & ".join(row)}\\\\')
    row = []
    for _ in range(2):
        row += ["$MSE$", "$R^2$", "$\\rho_{c}$"]
    print(f'& {" & ".join(row)}\\\\')
    order = VAD_ORDER if dataset == "vad" else PLUTCHIK_ORDER
    for embedding in order:
        embedding_path = EMBEDDINGS_REGRESSION[embedding]
        print(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}")
        for i, split in enumerate(REGRESSION_SPLITS):
            row = []
            for model in MODELS[:2]:
                path = os.path.join(result_path, f"{split}_{dataset}_{embedding_path}_{model}_train.parquet")
                df = pd.read_parquet(path)

                for metric in REGRESSION_METRICS:
                    scores = df.loc[metric+"_avg"].tolist()
                    row.append("$" + f"{np.mean(scores):.3f}"[1:] + "\\pm" + f"{np.std(scores):.3f}$"[1:])
            if i == 0:
                embedding_col = r"\multirow{2}{*}{" + embedding + "}"
            else:
                embedding_col = ""
            print(f'{embedding_col} & {" & ".join(row)}\\\\')

def regression2(dataset, result_path="results"):
    row = ["\\multicolumn{3}{c}{" + metric + "}" for metric in ["XGB", "MLP"]]
    print(f'& {" & ".join(row)}\\\\')
    row = []
    for _ in range(2):
        row += ["$MSE$", "$R^2$", "$\\rho_{c}$"]
    print(f'& {" & ".join(row)}\\\\')
    order = VAD_ORDER if dataset == "vad" else PLUTCHIK_ORDER
    for embedding in order:
        embedding_path = EMBEDDINGS_REGRESSION[embedding]
        print(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}")
        for i, split in enumerate(REGRESSION_SPLITS):
            row = []
            for model in MODELS[2:]:
                path = os.path.join(result_path, f"{split}_{dataset}_{embedding_path}_{model}_train.parquet")
                df = pd.read_parquet(path)

                for metric in REGRESSION_METRICS:
                    scores = df.loc[metric+"_avg"].tolist()
                    row.append("$" + f"{np.mean(scores):.3f}"[1:] + "\\pm" + f"{np.std(scores):.3f}$"[1:])
            if i == 0:
                embedding_col = r"\multirow{2}{*}{" + embedding + "}"
            else:
                embedding_col = ""
            print(f'{embedding_col} & {" & ".join(row)}\\\\')


def classification(result_path="results"):
    row = ["\\multicolumn{3}{c}{" + metric + "}" for metric in ["LR", "$k$-NN", "XGB", "MLP"]]
    print(f'& {" & ".join(row)}\\\\')
    row = []
    for _ in range(4):
        row += [r"\texttt{p}", r"\texttt{r}", r"$F_{1}$"]
    print(f'& {" & ".join(row)}\\\\')
    for embedding in GO_EMOTION_ORDER:
        embedding_path = EMBEDDINGS_CLASSIFICATION[embedding]
        print(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}")
        for i, split in enumerate(["macro", "weighted", "micro"]):
            row = []
            for model in MODELS:
                path = os.path.join(result_path, f"go_emotion_macro_{embedding_path}_{model}_train.parquet")
                df = pd.read_parquet(path)
                for metric in CLASSIFICATION_METRICS:
                    scores = df.loc[metric + f"_{split} avg"].tolist()
                    row.append("$" + f"{np.mean(scores):.3f}"[1:] + "\\pm" + f"{np.std(scores):.3f}$"[1:])
            if i == 0:
                embedding_col = r"\multirow{3}{*}{" + embedding + "}"
            else:
                embedding_col = ""
            print(f'{embedding_col} & {" & ".join(row)}\\\\')


if __name__ == '__main__':
    regression1("plutchik")
    print("\\\\")
    regression2("plutchik")
    #classification()
