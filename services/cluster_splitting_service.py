import os
import random
import sys
import numpy as np
import pandas as pd

from services import CACHED_PATH
from services.service import Service
from utils.dataset_utils import vad_mapping, plutchik_mapping


class ClusterSplittingService(Service):

    def __init__(self, method, cached_path=CACHED_PATH, n_folds=5, test_size=0.2, isolated_cluster_id=-1, excluded_cluster_id=-2, seed=42):
        filepath = os.path.join(cached_path, f'cluster_splits_{n_folds}fold_{method}.pkl')
        self.n_folds = n_folds
        self.test_size = test_size
        self.seed = seed
        self.isolated_cluster_id = isolated_cluster_id
        self.excluded_cluster_id = excluded_cluster_id
        self.words = []
        self.errors = []
        self.cluster_ids = []
        self.cluster_scores = []

        super().__init__(filepath)

    @staticmethod
    def randomize_clusters_by_size(clusters_by_size, rng):
        nested = []
        current = [clusters_by_size[0]]
        for i in range(1, len(clusters_by_size)):
            if clusters_by_size[i-1][0] == clusters_by_size[i][0]:
                current.append(clusters_by_size[i])
            else:
                if len(current) > 1:
                    rng.shuffle(current)
                nested.append(current)
                current = [clusters_by_size[i]]
        return [item for sublist in nested for item in sublist]


    def _create_splits(self, clustering_service, dictionary, alpha=0.5, seed=42):
        rng = random.Random(seed)

        scores = [dictionary[word] for word in clustering_service.all_phrases]

        cluster_scores = {}
        cluster_ids = {}

        for i, (cluster_id, word_score) in enumerate(zip(clustering_service.cluster_ids, scores)):
            if cluster_id ==  self.excluded_cluster_id:
                continue
            cluster_scores.setdefault(cluster_id, []).append(word_score)
            cluster_ids.setdefault(cluster_id, []).append(i)

        target_dist = np.concat(list(cluster_scores.values())).mean(axis=0)

        if self.isolated_cluster_id in cluster_ids:
            isolated = list(zip(cluster_ids.pop(self.isolated_cluster_id), cluster_scores.pop(self.isolated_cluster_id)))
            rng.shuffle(isolated)
        else:
            isolated = []

        cluster_by_size = [(len(ids), i) for i, ids in cluster_ids.items()]
        cluster_by_size.sort(reverse=True)

        # cluster_by_size = self.randomize_clusters_by_size(cluster_by_size, rng)

        valid_samples = sum(size for size, _ in cluster_by_size)

        n_bins = 1 + self.n_folds

        cluster_bin_ids = [cluster_ids[cluster_id] for _, cluster_id in cluster_by_size[:n_bins]]
        cluster_bin_scores = [np.array(cluster_scores[cluster_id]) for _, cluster_id in cluster_by_size[:n_bins]]
        total_size = sum(size for size, _ in cluster_by_size[:n_bins])

        target_ratios = [self.test_size] + [(1.0 - self.test_size) / self.n_folds] * self.n_folds
        target_sizes = [int(valid_samples * ratio) for ratio in target_ratios]

        for size, cluster_idx in cluster_by_size[n_bins:]:
            new_bin_ids = cluster_ids[cluster_idx]
            new_bin_scores = cluster_scores[cluster_idx]

            alpha_error = [np.linalg.norm(target_dist-bin_score.mean(axis=0)) for bin_score in cluster_bin_scores]
            alpha_updated_error = [np.linalg.norm(target_dist-np.concat((bin_score, new_bin_scores)).mean(axis=0)) for bin_score in cluster_bin_scores]
            alpha_change_error = np.array(alpha_updated_error) - np.array(alpha_error)

            beta_error = np.abs([ratio - (len(bin_ids)+size) / total_size for ratio, bin_ids in zip(target_ratios, cluster_bin_ids)])

            error = alpha*alpha_change_error + (1-alpha)*beta_error

            for bin_idx in np.argsort(error):
                if len(cluster_bin_ids[bin_idx])+size <= target_sizes[bin_idx]:
                    cluster_bin_scores[bin_idx] = np.concat((cluster_bin_scores[bin_idx], new_bin_scores))
                    cluster_bin_ids[bin_idx].extend(new_bin_ids)
                    break
            else:
                cluster_bin_scores[0] = np.concat((cluster_bin_scores[0], new_bin_scores))
                cluster_bin_ids[0].extend(new_bin_ids)
            total_size += size

        target_sizes = [int((valid_samples+len(isolated)) * ratio) for ratio in target_ratios]
        for isolated_idx, isolated_score in isolated:
            alpha_error = [np.linalg.norm(target_dist-bin_score.mean(axis=0)) for bin_score in cluster_bin_scores]
            alpha_updated_error = [np.linalg.norm(target_dist-np.concat((bin_score, [isolated_score])).mean(axis=0)) for bin_score in cluster_bin_scores]
            alpha_change_error = np.array(alpha_updated_error) - np.array(alpha_error)

            beta_error = np.abs([ratio - (len(bin_ids)+1) / total_size for ratio, bin_ids in zip(target_ratios, cluster_bin_ids)])

            error = alpha*alpha_change_error + (1-alpha)*beta_error

            for bin_idx in np.argsort(error):
                if len(cluster_bin_ids[bin_idx])+1 <= target_sizes[bin_idx]:
                    cluster_bin_scores[bin_idx] = np.concat((cluster_bin_scores[bin_idx], [isolated_score]))
                    cluster_bin_ids[bin_idx].append(isolated_idx)
                    break
            else:
                cluster_bin_scores[0] = np.concat((cluster_bin_scores[0], [isolated_score]))
                cluster_bin_ids[0].append(isolated_idx)
            total_size += 1

        words = [[clustering_service.all_phrases[id_] for id_ in ids] for ids in cluster_bin_ids]
        errors = [np.linalg.norm(target_dist-bin_score.mean(axis=0)) for bin_score in cluster_bin_scores]
        return errors, words, cluster_bin_ids, cluster_bin_scores

    def create_hyper_optimized_splits(self, clustering_service, dictionary, n_trails=100):
        rng = random.Random(self.seed)
        best_error = 100
        for i in range(n_trails):
            if i==0:
                alpha = 0.5
                seed = 42
            else:
                alpha = rng.betavariate(0.5, 0.5)
                seed = rng.randint(0, 2**30)
            result = self._create_splits(clustering_service, dictionary, alpha=alpha, seed=seed)
            errors, words, cluster_ids, cluster_scores = result
            error = np.sum(errors)
            if error < best_error:
                best_error = error
                self.errors, self.words, self.cluster_ids, self.cluster_scores = result
                print(i, best_error, alpha)
            else:
                print(i)

    def save(self):
        self._save((self.errors, self.words, self.cluster_ids, self.cluster_scores))

    def export_to_parquet(self, emotions, output_path=None):
        rng = random.Random(self.seed)
        data = []
        for i, (words, cluster_ids, cluster_scores) in enumerate(zip(self.words, self.cluster_ids, self.cluster_scores)):
            rows = []
            for word, cluster_id, cluster_score in zip(words, cluster_ids, cluster_scores):
                rows.append([word, i-1, cluster_id] + cluster_score.tolist())
            rng.shuffle(rows)
            data.extend(rows)

        columns = ["word", "fold_idx", "cluster_id"] + emotions
        df = pd.DataFrame(data, columns=columns)
        df.to_parquet(output_path)

    def load_from_cache(self):
        self.error, self.words, self.cluster_ids, self.cluster_scores = self._load_from_cache()


def save_morphological(dataset):
    from services.morphological_grouping_service import MorphologicalGroupingService
    from utils.dataset_utils import read_words_from_vad, read_words_from_plutchik

    morphological_service = MorphologicalGroupingService(dataset)
    morphological_service.load_from_cache()

    if dataset == 'vad':
        words_emotions = read_words_from_vad()
        emotion_columns = list(vad_mapping)
        n_trails = 50
    elif dataset == 'plutchik':
        words_emotions = read_words_from_plutchik()
        emotion_columns = list(plutchik_mapping)
        n_trails = 200
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Expected 'vad' or 'plutchik'")

    splitting_service = ClusterSplittingService(f"morphological_{dataset}")
    splitting_service.create_hyper_optimized_splits(morphological_service, words_emotions, n_trails)
    splitting_service.save()

    output_path = f"opendata/splits/cluster_splits_5fold_morphological_{dataset}.parquet"
    splitting_service.export_to_parquet(emotion_columns, output_path)


def save_community(dataset, edge_threshold):
    from services.community_clustering_service import CommunityClusteringService
    from utils.dataset_utils import read_words_from_vad, read_words_from_plutchik

    community_clustering = CommunityClusteringService(dataset, edge_threshold=edge_threshold, algorithm='leiden')
    community_clustering.load_from_cache()

    if dataset == 'vad':
        words_emotions = read_words_from_vad()
        emotion_columns = list(vad_mapping)
        n_trails = 50
    elif dataset == 'plutchik':
        words_emotions = read_words_from_plutchik()
        emotion_columns = list(plutchik_mapping)
        n_trails = 200
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Expected 'vad' or 'plutchik'")

    splitting_service = ClusterSplittingService(f"community_{dataset}")
    splitting_service.create_hyper_optimized_splits(community_clustering, words_emotions, n_trails)
    splitting_service.save()

    output_path = f"opendata/splits/cluster_splits_5fold_community_{dataset}.parquet"
    splitting_service.export_to_parquet(emotion_columns, output_path)


def load(n_folds=5, random_state=42):
    """Load splits from cache."""
    service = ClusterSplittingService(
        method='morphological',  # Placeholder, will be overridden by cache
        n_folds=n_folds,
        seed=random_state
    )
    service.load_from_cache()

    print(f"\nTotal samples: {service.n_samples}", file=sys.stderr)
    print(f"Number of folds: {service.n_folds}", file=sys.stderr)
    print(f"Test set size: {len(service.splits['test'])}", file=sys.stderr)

    # Show example fold
    fold_data = service.get_fold_data(0)
    print(f"\nFold 1 example:", file=sys.stderr)
    print(f"  Train size: {len(fold_data['train'])}", file=sys.stderr)
    print(f"  Val size: {len(fold_data['val'])}", file=sys.stderr)

    print('\nFinished!')
    return service


if __name__ == '__main__':
    # save_morphological("vad")
    save_community("vad", 0.9)
    # save_morphological("plutchik")
    # save_community("plutchik", 0.88)
