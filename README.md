# A Comparative Study on Affective Cues in Text Embeddings Across Psychological Emotion Theories

This is the repository for the respective paper, providing all results and details about the implementation.

## Embeddings

[embedding_service.py](services/embedding_service.py) calculates the embeddings of open-weight models.

The [`embeddings`](embeddings) directory contains the embeddings for all
open-weight and closed models across all three datasets, stored via Git LFS.
These embeddings are provided solely to enable reproduction of our results;
any further use is subject to the respective license of each individual model provider. 
In particular, embeddings derived from non-commercially licensed models may not be used for commercial purposes.


## Predictive Models (Training and Testing)

* [regression_benchmark_service.py](services/regression_benchmark_service.py)
provides the experimental setup, i.e. predictive model configuration & hyperparameter search space
for the NRC-VAD and NRC-EIL.

* [classification_benchmark_service.py](services/classification_benchmark_service.py) 
provides the experimental setup, i.e. predictive model configuration & hyperparameter search space
for the GoEmotion classification task.

* [schedule_regression.sh](schedule_regression.sh) provides code for training and testing the regression
models on CPU/GPU clusters. Here the actual program parameters for
[regression_benchmark_service.py](services/regression_benchmark_service.py)
can be found.

* [schedule_classification.sh](schedule_classification.sh) provides code for training and testing the classification 
models on CPU/GPU clusters. Here the actual program parameters for
[classification_benchmark_service.py](services/classification_benchmark_service.py)
can be found.

The training and test results are provided [here](data/results) together with the predictions for the test sets.
Collected data during hyperparameter search are stored within the [optuna](cached/optuna) directory.


## Data Splitting

* [morphological_grouping_service.py](services/morphological_grouping_service.py) groups NRC-VAD and NRC-EIL into
morphological groups.

* [community_clustering_service.py](services/community_clustering_service.py) uses the similarity matrix of
[wordnet_similarity_service.py](services/wordnet_similarity_service.py) to cluster the words of NRC-VAD and NRC-EIL.

* [cluster_splitting_service.py](services/cluster_splitting_service.py) creates for NRC-VAD and NRC-EIL 
5-fold cv and holdout-splits. This services balances emotional labels. 
Input can be the groups of [morphological_grouping_service.py](services/morphological_grouping_service.py) as well as
the clusters of [community_clustering_service.py](services/community_clustering_service.py)

[go_emotion_service.py](services/go_emotion_service.py) merges the predefined `train` and `dev` set and creates a
5-fold cv split. For evaluation the predefined `test` set is used.

The created splits are available in [opendata](data/opendata) and can be used for future research.

