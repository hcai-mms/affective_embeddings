import os
import sys
import numpy as np
from multiprocessing import Pool, cpu_count
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

from services import CACHED_PATH
from services.service import Service


def compute_wordnet_similarity(word1, word2):
    def get_synsets(word):
        synsets = []
        for pos in [wn.NOUN, wn.VERB, wn.ADJ, wn.ADV]:
            synsets.extend(wn.synsets(word, pos=pos))
        return synsets

    synsets1 = get_synsets(word1)
    synsets2 = get_synsets(word2)

    if not synsets1 or not synsets2:
        print("Ohhh noooOoOo! that should never happen!")
        return 0.0

    max_sim = 0.0
    for s1 in synsets1:
        for s2 in synsets2:
            sim = s1.wup_similarity(s2)
            if sim is not None and sim > max_sim:
                max_sim = sim

    return max_sim


def compute_similarity_row(args):
    i, word_i, all_words, start_j = args
    n = len(all_words)
    row = np.zeros(n)

    for j in range(start_j, n):
        if i == j:
            row[j] = 1.0
        else:
            row[j] = compute_wordnet_similarity(word_i, all_words[j])

    return i, row


class WordNetSimilarityService(Service):
    def __init__(self, dataset, cached_path=CACHED_PATH):
        filepath = os.path.join(cached_path, f'wordnet_similarity_matrix_{dataset}.pkl')

        self.all_phrases = []
        self.content_words = []
        self.unique_content_words = []
        self.word_mapping = {}
        self.similarity_matrix = None
        self.coverage_stats = {}
        self._init_nltk_resources()

        super().__init__(filepath)

    def _init_nltk_resources(self):
        try:
            wn.synsets('test')
        except LookupError:
            print("Downloading WordNet...", file=sys.stderr)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)

        try:
            self.stopwords = set(stopwords.words('english'))
        except LookupError:
            print("Downloading stopwords...", file=sys.stderr)
            nltk.download('stopwords', quiet=True)
            self.stopwords = set(stopwords.words('english'))

    def has_synsets(self, word):
        for pos in [wn.NOUN, wn.VERB, wn.ADJ, wn.ADV]:
            if wn.synsets(word, pos=pos):
                return True
        return False

    def get_content_word(self, phrase):
        words = phrase.lower().split()

        # If multi-word phrase, try it in WordNet with underscores
        if len(words) > 1:
            wordnet_phrase = '_'.join(words)
            if self.has_synsets(wordnet_phrase):
                return wordnet_phrase

        # Fall back to first content word
        content_words = [w for w in words if w not in self.stopwords]

        # If all words are stopwords, use the first word
        if not content_words:
            return words[0]

        return content_words[0]

    def build_similarity_matrix(self, content_words, checkpoint_interval=1000):
        n = len(content_words)
        similarity_matrix = np.zeros((n, n))

        checkpoint_file = os.path.join(CACHED_PATH, f'similarity_matrix_checkpoint_{n}words.npz')
        start_idx = 0

        if os.path.exists(checkpoint_file):
            print(f"Found checkpoint file, loading...", file=sys.stderr)
            checkpoint_data = np.load(checkpoint_file)
            similarity_matrix = checkpoint_data['similarity_matrix']
            start_idx = int(checkpoint_data['completed_rows'])
            print(f"Resuming from row {start_idx}/{n}", file=sys.stderr)

        print(f"\nComputing WordNet similarities for {n} unique words...", file=sys.stderr)
        print(f"Checkpointing every {checkpoint_interval} rows", file=sys.stderr)

        tasks = []
        for i in range(start_idx, n):
            tasks.append((i, content_words[i], content_words, i))

        with Pool(processes=cpu_count()) as pool:
            results = []
            rows_processed = 0

            for idx, result in enumerate(pool.imap_unordered(compute_similarity_row, tasks)):
                i, row = result
                results.append((i, row))
                rows_processed += 1

                similarity_matrix[i, :] = row

                if (rows_processed) % 100 == 0:
                    print(f"  Processed {start_idx + rows_processed}/{n} words...", file=sys.stderr)

                if (rows_processed) % checkpoint_interval == 0:
                    completed_so_far = start_idx + rows_processed
                    print(f"  Saving checkpoint at {completed_so_far}/{n} rows...", file=sys.stderr)
                    np.savez_compressed(
                        checkpoint_file,
                        similarity_matrix=similarity_matrix,
                        completed_rows=completed_so_far
                    )

        print(f"  Completed all {n} words!", file=sys.stderr)
        print(f"  Saving final checkpoint...", file=sys.stderr)
        np.savez_compressed(
            checkpoint_file,
            similarity_matrix=similarity_matrix,
            completed_rows=n
        )
        print(f"  Checkpoint saved to {checkpoint_file}", file=sys.stderr)

        return similarity_matrix

    def load_from_data(self, words):
        print(f"Loading {len(words)} words/phrases...", file=sys.stderr)

        self.all_phrases = list(words)
        self.content_words = []

        multiword_count = 0
        multiword_in_wordnet = 0

        for phrase in self.all_phrases:
            # Track multi-word phrases
            is_multiword = len(phrase.split()) > 1
            if is_multiword:
                multiword_count += 1
                # Check if found in WordNet
                wordnet_phrase = '_'.join(phrase.lower().split())
                if self.has_synsets(wordnet_phrase):
                    multiword_in_wordnet += 1

            content_word = self.get_content_word(phrase)
            self.content_words.append(content_word)
            self.word_mapping[phrase] = content_word

        print(f"Loaded {len(self.all_phrases)} words/phrases", file=sys.stderr)
        print(f"  Multi-word phrases: {multiword_count}", file=sys.stderr)
        if multiword_count > 0:
            print(f"  Multi-word found in WordNet: {multiword_in_wordnet} ({100*multiword_in_wordnet/multiword_count:.1f}%)", file=sys.stderr)
        else:
            print(f"  Multi-word found in WordNet: 0 (0.0%)", file=sys.stderr)

        self.compute_and_store_similarities(multiword_count, multiword_in_wordnet)

        return self


    def compute_and_store_similarities(self, multiword_count, multiword_in_wordnet):
        self.unique_content_words = list(set(self.content_words))
        n_unique = len(self.unique_content_words)

        print(f"\nPrefiltering words by WordNet coverage...", file=sys.stderr)
        words_with_synsets = []
        words_without_synsets = []

        for word in self.unique_content_words:
            if self.has_synsets(word):
                words_with_synsets.append(word)
            else:
                words_without_synsets.append(word)

        print(f"  Total unique content words: {n_unique}", file=sys.stderr)
        print(f"  Words WITH synsets: {len(words_with_synsets)} ({100*len(words_with_synsets)/n_unique:.1f}%)", file=sys.stderr)
        print(f"  Words WITHOUT synsets: {len(words_without_synsets)} ({100*len(words_without_synsets)/n_unique:.1f}%)", file=sys.stderr)

        if words_without_synsets:
            print(f"\n  Example words without synsets: {words_without_synsets[:10]}", file=sys.stderr)

        if len(words_with_synsets) == 0:
            raise ValueError("No words have WordNet synsets! Cannot proceed with similarity computation.")

        # Store coverage statistics
        self.coverage_stats = {
            'total_phrases': len(self.all_phrases),
            'total_unique_content_words': n_unique,
            'words_with_synsets': len(words_with_synsets),
            'words_without_synsets': len(words_without_synsets),
            'multiword_count': multiword_count,
            'multiword_in_wordnet': multiword_in_wordnet,
            'words_with_synsets_list': words_with_synsets,
            'words_without_synsets_list': words_without_synsets
        }

        print(f"\nComputing similarity matrix for {len(words_with_synsets)} words with synsets...", file=sys.stderr)
        similarity_matrix_partial = self.build_similarity_matrix(words_with_synsets)

        # Create full similarity matrix (including words without synsets)
        # Words without synsets get 0 similarity to everything except themselves
        word_to_idx = {word: idx for idx, word in enumerate(self.unique_content_words)}
        n_all = len(self.unique_content_words)
        self.similarity_matrix = np.zeros((n_all, n_all))

        # Fill in similarities for words with synsets
        for i, word_i in enumerate(words_with_synsets):
            idx_i = word_to_idx[word_i]
            for j, word_j in enumerate(words_with_synsets):
                idx_j = word_to_idx[word_j]
                self.similarity_matrix[idx_i, idx_j] = similarity_matrix_partial[i, j]

        # Words without synsets: only diagonal is 1.0 (self-similarity)
        for word in words_without_synsets:
            idx = word_to_idx[word]
            self.similarity_matrix[idx, idx] = 1.0

        # Symmetrize the matrix by copying upper triangle to lower triangle
        print(f"\nSymmetrizing similarity matrix...", file=sys.stderr)
        self.similarity_matrix = self.similarity_matrix + self.similarity_matrix.T - np.diag(np.diag(self.similarity_matrix))

        print(f"Similarity matrix computed: {self.similarity_matrix.shape}", file=sys.stderr)

    def save(self):
        data = {
            'similarity_matrix': self.similarity_matrix,
            'unique_content_words': self.unique_content_words,
            'all_phrases': self.all_phrases,
            'content_words': self.content_words,
            'word_mapping': self.word_mapping,
            'coverage_stats': self.coverage_stats
        }

        print(f"\nSaving similarity matrix and metadata...", file=sys.stderr)
        print(f"  Matrix shape: {self.similarity_matrix.shape}", file=sys.stderr)
        print(f"  Total phrases: {len(self.all_phrases)}", file=sys.stderr)
        print(f"  Unique content words: {len(self.unique_content_words)}", file=sys.stderr)

        self._save(data)
        print("Done!", file=sys.stderr)

    def load_from_cache(self):
        data = self._load_from_cache()
        self.similarity_matrix = data['similarity_matrix']
        self.unique_content_words = data['unique_content_words']
        self.all_phrases = data['all_phrases']
        self.content_words = data['content_words']
        self.word_mapping = data['word_mapping']
        self.coverage_stats = data['coverage_stats']

        print(f"Loaded similarity matrix from cache", file=sys.stderr)
        print(f"  Matrix shape: {self.similarity_matrix.shape}", file=sys.stderr)
        print(f"  Total phrases: {len(self.all_phrases)}", file=sys.stderr)
        print(f"  Unique content words: {len(self.unique_content_words)}", file=sys.stderr)


def save_vad():
    from utils.dataset_utils import read_words_from_vad

    print(f"Loading words from vad...", file=sys.stderr)
    words = list(read_words_from_vad())
    print(f"Loaded {len(words)} words/phrases from vad", file=sys.stderr)

    service = WordNetSimilarityService("vad")
    service.load_from_data(words)
    service.save()


def save_plutchik():
    from utils.dataset_utils import read_words_from_plutchik

    print(f"Loading words from plutchik...", file=sys.stderr)
    words = list(read_words_from_plutchik())
    print(f"Loaded {len(words)} words/phrases from plutchik", file=sys.stderr)

    service = WordNetSimilarityService("plutchik")
    service.load_from_data(words)
    service.save()


def load(dataset):
    service = WordNetSimilarityService(dataset)
    service.load_from_cache()

    stats = service.coverage_stats
    print(f"\nCoverage Statistics:", file=sys.stderr)
    print(f"  Total phrases: {stats['total_phrases']}", file=sys.stderr)
    print(f"  Unique content words: {stats['total_unique_content_words']}", file=sys.stderr)
    print(f"  Words with synsets: {stats['words_with_synsets']} ({100*stats['words_with_synsets']/stats['total_unique_content_words']:.1f}%)", file=sys.stderr)
    print(f"  Multi-word phrases: {stats['multiword_count']}", file=sys.stderr)
    if stats['multiword_count'] > 0:
        print(f"  Multi-word in WordNet: {stats['multiword_in_wordnet']} ({100*stats['multiword_in_wordnet']/stats['multiword_count']:.1f}%)", file=sys.stderr)

    print('\nFinished!')


if __name__ == '__main__':
    #save_vad()
    save_plutchik()
    #load()
