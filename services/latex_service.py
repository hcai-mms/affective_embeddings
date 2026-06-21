from services.significance_service import SignificanceService

MODELS = ["elasticnet", "knn", "xgboost", "mlp"]
REGRESSION_SPLITS = ["community", "morphological", ]
REGRESSION_DATASETS = ["vad", "plutchik"]
CLASSIFICATION_SPLITS = ["macro", "weighted", "micro"]

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

REGRESSION_METRICS = ["mse", "r2", "ccc"]
CLASSIFICATION_METRICS = ["precision", "recall", "f1"]

def regression(dataset):
    significance_service = SignificanceService(dataset)
    significance_service.load_from_cache()

    _, mlp_order = significance_service.mapping[("community", "mlp", "r2")]

    # row = [r"\multicolumn{3}{c}{" + metric + "}" for metric in ["LR", "$k$-NN", "XGB", "MLP"]]
    # print(f'& {" & ".join(row)}\\\\')
    # row = []
    # for _ in range(4):
    #     row += ["$MSE$", "$R^2$", "$\\rho_{c}$"]
    # print(f'& {" & ".join(row)}\\\\')
    for _, embedding in mlp_order:
        row = []
        for model in MODELS:
            for metric in REGRESSION_METRICS:
                key = "community", model, metric
                scores, order = significance_service.mapping[key]
                score, p_value = scores[embedding]
                best_model = order[0][1]
                if best_model == embedding:
                    row.append("$\\mathbf{" + f"{score:.3f}"[1:] + "}$")
                else:
                    if p_value > 0.05:
                        row.append("$\\underline{\\underline{"+ f"{score:.3f}"[1:] + "}}$")
                    elif p_value > 0.005:
                        row.append("$\\underline{" + f"{score:.3f}"[1:] + "}$")
                    else:
                        row.append("$" + f"{score:.3f}"[1:] + "$")
        embedding_col = r"\textbf{" + embedding + "}"
        print(f'{embedding_col} & {" & ".join(row)}\\\\')


def classification():
    # row = ["\\multicolumn{3}{c}{" + metric + "}" for metric in ["LR", "$k$-NN", "XGB", "MLP"]]
    # print(f'& {" & ".join(row)}\\\\')
    # row = []
    # for _ in range(4):
    #     row += [r"\texttt{p}", r"\texttt{r}", r"$F_{1}$"]
    # print(f'& {" & ".join(row)}\\\\')
    significance_service = SignificanceService("go_emotion")
    significance_service.load_from_cache()

    _, mlp_order = significance_service.mapping[("macro", "mlp", "f1")]

    for _, embedding in mlp_order:
        row = []
        for model in MODELS:
            for metric in CLASSIFICATION_METRICS:
                key = "macro", model, metric
                scores, order = significance_service.mapping[key]
                score, p_value = scores[embedding]
                best_model = order[0][1]
                if best_model == embedding:
                    row.append("$\\mathbf{" + f"{score:.3f}"[1:] + "}$")
                else:
                    if p_value > 0.05:
                        row.append("$\\underline{\\underline{"+ f"{score:.3f}"[1:] + "}}$")
                    elif p_value > 0.005:
                        row.append("$\\underline{" + f"{score:.3f}"[1:] + "}$")
                    else:
                        row.append("$" + f"{score:.3f}"[1:] + "$")
        embedding_col = r"\textbf{" + embedding + "}"
        print(f'{embedding_col} & {" & ".join(row)}\\\\')


if __name__ == '__main__':
    # regression("vad")
    # regression("plutchik")
    classification()
