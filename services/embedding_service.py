import gc
import os
import sys

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from utils.dataset_utils import get_corpus

MODEL_CONFIGS = {
    "google@embeddinggemma-300m": {
        "hf_name": "google/embeddinggemma-300m",
        "model_kwargs": {},
        "encode_kwargs": {
            "batch_size": 128,
            "prompt_name": "Classification",
        }
        # https://arxiv.org/pdf/2509.20354
        # better use prompt name
    },
    "google@embeddinggemma-300m(STS": {
        "hf_name": "google/embeddinggemma-300m",
        "model_kwargs": {},
        "encode_kwargs": {
            "batch_size": 128,
            "prompt_name": "STS",
        }
        # https://arxiv.org/pdf/2509.20354
        # better use prompt name
    },
    "sentence-transformers@sentence-t5-xxl": {
        "hf_name": "sentence-transformers/sentence-t5-xxl",
        "model_kwargs": {},
        "encode_kwargs": {
            "batch_size": 2,
            "convert_to_numpy": True,
        }
        # https://arxiv.org/pdf/2108.08877
        # no prompt
    },
    "nomic-ai@nomic-embed-text-v2-moe": {
        "hf_name": "nomic-ai/nomic-embed-text-v2-moe",
        "model_kwargs": {"trust_remote_code": True},
        "encode_kwargs": {
            "batch_size": 64,
            "prompt_name": "Classification",
        }
        # https://arxiv.org/pdf/2502.07972
    },
    "jinaai@jina-embeddings-v4": {
        "hf_name": "jinaai/jina-embeddings-v4",
        "model_kwargs": {},
        "encode_kwargs": {
            "batch_size": 32,
            "task": "text-matching"
        }
    },
    "tencent@KaLM-Embedding-Gemma3-12B-2511": {
        "hf_name": "tencent/KaLM-Embedding-Gemma3-12B-2511",
        "model_kwargs": {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2"
        },
        "encode_kwargs": {
            "batch_size": 1,
            "prompt": "Instruct: Classifying emotion.\nQuery:"
        },
        # https://arxiv.org/pdf/2506.20923
        # https://huggingface.co/tencent/KaLM-Embedding-Gemma3-12B-2511
    },
    "tencent@KaLM-Embedding-Gemma3-12B-2511(retrieval)": {
        "hf_name": "tencent/KaLM-Embedding-Gemma3-12B-2511",
        "model_kwargs": {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2"
        },
        "encode_kwargs": {
            "batch_size": 1,
            "prompt": "Instruct: Retrieving emotion.\nQuery:"
        },
        # https://arxiv.org/pdf/2506.20923
        # https://huggingface.co/tencent/KaLM-Embedding-Gemma3-12B-2511
    },
    "intfloat@multilingual-e5-large-instruct": {
        "hf_name": "intfloat/multilingual-e5-large-instruct",
        "model_kwargs": {},
        "encode_kwargs": {
            "batch_size": 8,
            "prompt": "Instruct: Classify the emotion expressed in the text\nQuery: ",
            "normalize_embeddings": True
        },
        # https://huggingface.co/intfloat/multilingual-e5-large-instruct
    },
    "intfloat@multilingual-e5-large-instruct(retrieval)": {
        "hf_name": "intfloat/multilingual-e5-large-instruct",
        "model_kwargs": {},
        "encode_kwargs": {
            "batch_size": 8,
            "prompt": "Instruct: Retrieve the emotion expressed in the text\nQuery: ",
            "normalize_embeddings": True
        },
        # https://huggingface.co/intfloat/multilingual-e5-large-instruct
    },
    "nvidia@llama-embed-nemotron-8b": {
        "hf_name": "nvidia/llama-embed-nemotron-8b",
        "model_kwargs": {
            "attn_implementation": "eager",
            "torch_dtype": "bfloat16"
        },
        "tokenizer_kwargs": {"padding_side": "left"},
        "encode_kwargs": {
            "batch_size": 2,
            "prompt": "Instruct: Classify the emotion expressed in the text\nQuery: "
        },
    },
    "nvidia@llama-embed-nemotron-8b(retrieval)": {
        "hf_name": "nvidia/llama-embed-nemotron-8b",
        "model_kwargs": {
            "attn_implementation": "eager",
            "torch_dtype": "bfloat16"
        },
        "tokenizer_kwargs": {"padding_side": "left"},
        "encode_kwargs": {
            "batch_size": 2,
            "prompt": "Instruct: Retrieve the emotion expressed in the text\nQuery: "
        },
    },
    "Qwen@Qwen3-Embedding-8B": {
        "hf_name": "Qwen/Qwen3-Embedding-8B",
        "model_kwargs": {},
        "encode_kwargs": {
            "batch_size": 2,
            "prompt": "Instruct: Classify the emotion expressed in the text\nQuery:",
        }
    },
    "Qwen@Qwen3-Embedding-8B(retrieval)": {
        "hf_name": "Qwen/Qwen3-Embedding-8B",
        "model_kwargs": {},
        "encode_kwargs": {
            "batch_size": 2,
            "prompt": "Instruct: Retrieve the emotion expressed in the text\nQuery:",
        }
    },
    "Linq-AI-Research@Linq-Embed-Mistral": {
        "hf_name": "Linq-AI-Research/Linq-Embed-Mistral",
        "model_kwargs": {},
        "encode_kwargs": {
            "batch_size": 2,
            "prompt": "Instruct: Classify the emotion expressed in the text\nQuery: "
        }
    },
    "Linq-AI-Research@Linq-Embed-Mistral(retrieval)": {
        "hf_name": "Linq-AI-Research/Linq-Embed-Mistral",
        "model_kwargs": {},
        "encode_kwargs": {
            "batch_size": 2,
            "prompt": "Instruct: Retrieve the emotion expressed in the text\nQuery: "
        }
    }
}


class EmbeddingService:

    def __init__(self, model_name, embedding_path: str = "embeddings"):
        self.model_name = model_name
        self.embedding_path = os.path.join(embedding_path, f'embeddings_{model_name}.parquet')
        self.embeddings: dict[str, np.ndarray] = {}
        self.hf_name = ""
        self.encode_method = ""
        self.constructor_kwargs = {}
        self.encode_kwargs = {"batch_size": 64, "show_progress_bar": True}
        self.set_config()

    def set_config(self):
        if self.model_name not in MODEL_CONFIGS:
            raise ValueError(f"Model '{self.model_name}' not found in MODEL_CONFIGS")

        config = MODEL_CONFIGS[self.model_name].copy()
        
        self.hf_name = config.pop("hf_name")
        self.encode_method = config.pop("encode_method", "encode")
        encode_kwargs = config.pop("encode_kwargs", {})

        self.constructor_kwargs = config
        self.encode_kwargs.update(encode_kwargs)

    def generate_embeddings(self, corpus: list[str]) -> dict[str, np.ndarray]:
        not_found = list(set(corpus).difference(self.embeddings))
        if len(not_found) == 0:
            raise ValueError("Already calculated all embeddings")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for embeddings")

        print(f"Generating embeddings: {len(corpus)} items, model: {self.model_name}", file=sys.stderr)
        print(f"Device: cuda ({torch.cuda.get_device_name(0)})", file=sys.stderr)

        model = SentenceTransformer(
            self.hf_name,
            device="cuda",
            trust_remote_code=True,
            **self.constructor_kwargs
        )

        encode_fn = getattr(model, self.encode_method)
        embeddings_array = encode_fn(not_found, **self.encode_kwargs)

        if not isinstance(embeddings_array, np.ndarray):
            embeddings_array = np.array(embeddings_array)

        print(f"Generated {len(embeddings_array)} embeddings", file=sys.stderr)

        self.embeddings.update({word: embeddings_array[i] for i, word in enumerate(not_found)})

        del model
        gc.collect()
        torch.cuda.empty_cache()

        return self.embeddings

    def save(self, embeddings: dict[str, np.ndarray] | None = None):
        if embeddings is None:
            embeddings = self.embeddings

        print(f"Saving {len(embeddings)} embeddings to {self.embedding_path}", file=sys.stderr)
        df = pd.DataFrame.from_dict(embeddings, orient='index', dtype=np.float32)
        df.to_parquet(self.embedding_path)
        print("Done!", file=sys.stderr)

    def load(self) -> dict[str, np.ndarray]:
        """Load embeddings from parquet cache."""
        if not os.path.exists(self.embedding_path):
            raise FileNotFoundError(f"No cached embeddings: {self.embedding_path}")

        print(f"Loading embeddings from {self.embedding_path}", file=sys.stderr)
        df = pd.read_parquet(self.embedding_path)

        self.embeddings = {
            word: np.array(row.values, dtype=np.float32)
            for word, row in df.iterrows()
        }
        print(f"Loaded {len(self.embeddings)} embeddings", file=sys.stderr)
        return self.embeddings


def generate_and_save(model_name: str, corpus: list[str] | None = None):
    """Generate and save embeddings for a model."""
    corpus = corpus or get_corpus()
    service = EmbeddingService(model_name=model_name)

    if os.path.exists(service.embedding_path):
        service.load()

    embeddings = service.generate_embeddings(corpus)
    service.save(embeddings)


def generate_all(corpus: list[str] | None = None):
    """Generate and save embeddings for all configured models."""
    corpus = corpus or get_corpus()

    for model_name in MODEL_CONFIGS:
        print(f"\n{'=' * 80}", file=sys.stderr)
        print(f"Processing: {model_name}", file=sys.stderr)
        print(f"{'=' * 80}", file=sys.stderr)

        try:
            generate_and_save(model_name, corpus)
        except Exception as e:
            print(f"ERROR: {model_name} failed: {e}", file=sys.stderr)

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    generate_all()
