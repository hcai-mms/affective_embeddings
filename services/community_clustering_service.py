import os
import sys
import numpy as np
from collections import defaultdict
import igraph as ig
import leidenalg

from services import CACHED_PATH
from services.service import Service
from services.wordnet_similarity_service import WordNetSimilarityService


class CommunityClusteringService(Service):
    def __init__(self, dataset, cached_path=CACHED_PATH, edge_threshold=0.3, algorithm='leiden', random_state=42):
        filepath = os.path.join(
            cached_path,
            f'{algorithm}_clusters_threshold{edge_threshold}_{dataset}.pkl'
        )
        self.edge_threshold = edge_threshold
        self.algorithm = algorithm
        self.random_state = random_state

        self.all_phrases = []
        self.cluster_ids = []

        super().__init__(filepath)

    def load_from_data(self, similarity_service: WordNetSimilarityService):
        self.all_phrases = similarity_service.all_phrases
        content_words = similarity_service.content_words
        unique_content_words = similarity_service.unique_content_words
        similarity_matrix = similarity_service.similarity_matrix
        words_not_in_wordnet = set(similarity_service.coverage_stats['words_without_synsets_list'])

        print(f"  Building weighted graph...", file=sys.stderr)

        # Create adjacency matrix with threshold applied
        adjacency = (similarity_matrix >= self.edge_threshold).astype(float)
        np.fill_diagonal(adjacency, 0)
        weighted_adjacency = adjacency * similarity_matrix

        G = ig.Graph.Weighted_Adjacency(
            weighted_adjacency.tolist(),
            mode='upper',  # Upper triangular (undirected graph)
            attr='weight',
            loops=False
        )

        # Identify isolated nodes (no edges)
        degrees = G.degree()
        isolated_mask = np.array(degrees) == 0
        isolated_indices = np.where(isolated_mask)[0]

        np.random.seed(self.random_state)

        if self.algorithm == 'leiden':
            partition = leidenalg.find_partition(
                G,
                leidenalg.RBConfigurationVertexPartition,
                weights='weight',
                seed=self.random_state
            )
        elif self.algorithm == 'louvain':
            partition = G.community_multilevel(
                weights='weight',
                return_levels=False
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}. Use 'leiden' or 'louvain'")

        print(f"  Modularity: {partition.modularity:.4f}", file=sys.stderr)

        cluster_labels = np.array(partition.membership)

        for idx in isolated_indices:
            word = unique_content_words[idx]
            if word in words_not_in_wordnet:
                cluster_labels[idx] = -2
            else:
                cluster_labels[idx] = -1

        word_to_cluster = {word: cluster_id for word, cluster_id in zip(unique_content_words, cluster_labels)}

        self.cluster_ids = [word_to_cluster[w] for w in content_words]
        self.show_example_clusters()

    def show_example_clusters(self, top=5):
        print("\nExample semantic clusters:", file=sys.stderr)

        cluster_to_words = defaultdict(list)
        for phrase, cluster_id in zip(self.all_phrases, self.cluster_ids):
            if cluster_id >= 0:  # Exclude isolated nodes
                cluster_to_words[cluster_id].append(phrase)

        interesting_clusters = [cid for cid, words in cluster_to_words.items() if len(words) >= 3][:top]

        for cluster_id in interesting_clusters:
            words = cluster_to_words[cluster_id][:8]
            print(f"  Cluster {cluster_id}: {words}", file=sys.stderr)

    def save(self):
        print(f"\nSaving clusters for {len(self.all_phrases)} phrases...", file=sys.stderr)
        self._save((self.all_phrases, self.cluster_ids))
        print("Done!", file=sys.stderr)

    def load_from_cache(self):
        self.all_phrases, self.cluster_ids = self._load_from_cache()
        print(f"Loaded clusters for {len(self.all_phrases)} phrases from cache", file=sys.stderr)

    def get_all_clusters(self):
        clusters = defaultdict(list)
        for phrase, cluster_id in zip(self.all_phrases, self.cluster_ids):
            clusters[cluster_id].append(phrase)

        return dict(clusters)


def save(dataset, edge_threshold, algorithm='leiden', random_state=42):
    similarity_service = WordNetSimilarityService(dataset)
    similarity_service.load_from_cache()

    service = CommunityClusteringService(
        dataset,
        edge_threshold=edge_threshold,
        algorithm=algorithm,
        random_state=random_state
    )
    service.load_from_data(similarity_service)
    service.save()


def load(dataset, edge_threshold=0.3, algorithm='leiden', random_state=42):
    service = CommunityClusteringService(
        dataset,
        edge_threshold=edge_threshold,
        algorithm=algorithm,
        random_state=random_state
    )
    service.load_from_cache()
    service.show_example_clusters(5)
    print('\nFinished!')

    return service


if __name__ == '__main__':
    # save("plutchik", edge_threshold=0.88, algorithm='leiden')
    save("vad", edge_threshold=0.9, algorithm='leiden')
    # load("plutchik", edge_threshold=0.88, algorithm='leiden')
    # load("vad", edge_threshold=0.9, algorithm='leiden')
