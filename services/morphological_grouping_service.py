import os
import sys
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

from services import CACHED_PATH
from services.service import Service
from utils.dataset_utils import read_words_from_vad, read_words_from_plutchik


class MorphologicalGroupingService(Service):
    def __init__(self, dataset, cached_path=CACHED_PATH, random_state=42):
        filepath = os.path.join(cached_path, f'morphological_groups_{dataset}.pkl')
        self.random_state = random_state

        self.all_phrases = []
        self.cluster_ids = []
        self.stemmer = SnowballStemmer('english')

        try:
            self.stopwords = set(stopwords.words('english'))
        except LookupError:
            print("Downloading stopwords...", file=sys.stderr)
            nltk.download('stopwords', quiet=True)
            self.stopwords = set(stopwords.words('english'))

        super().__init__(filepath)

    # This is not AI, it is just useful here to have a description
    def get_lemma(self, word):
        words = word.lower().split()

        # Filter out stopwords, keep only content words
        content_words = [w for w in words if w not in self.stopwords]

        # If all words were stopwords, keep at least the original phrase
        if not content_words:
            content_words = words

        # Stem each content word
        stemmed_words = [self.stemmer.stem(w) for w in content_words]

        # Join with underscore to create compound stem
        return '_'.join(stemmed_words)

    def load_from_data(self, words):
        print(f"Creating morphological groups from {len(words)} words...", file=sys.stderr)

        self.all_phrases = list(words)

        lemma_to_indices = {}
        for idx, phrase in enumerate(self.all_phrases):
            lemma = self.get_lemma(phrase)
            lemma_to_indices.setdefault(lemma, []).append(idx)

        unique_groups = sorted(lemma_to_indices.keys())
        lemma_to_group_id = {lemma: gid for gid, lemma in enumerate(unique_groups)}

        self.cluster_ids = [0] * len(self.all_phrases)
        for lemma, indices in lemma_to_indices.items():
            group_id = lemma_to_group_id[lemma]
            for idx in indices:
                self.cluster_ids[idx] = group_id

        print(f"Found {len(unique_groups)} unique morphological groups", file=sys.stderr)
        self.show_example_groups(lemma_to_indices)

    def show_example_groups(self, lemma_to_indices, top=5):
        print("\nExample morphological groups:", file=sys.stderr)
        multi_member_groups = [(lemma, indices) for lemma, indices in lemma_to_indices.items() if len(indices) > 1]

        for lemma, indices in sorted(multi_member_groups, key=lambda x: len(x[1]), reverse=True)[:top]:
            words_in_group = [self.all_phrases[i] for i in indices[:5]]
            print(f"  {lemma}: {words_in_group}", file=sys.stderr)

    def save(self):
        print(f"\nSaving morphological groups for {len(self.all_phrases)} phrases...", file=sys.stderr)
        self._save((self.all_phrases, self.cluster_ids))
        print("Done!", file=sys.stderr)

    def load_from_cache(self):
        self.all_phrases, self.cluster_ids = self._load_from_cache()
        print(f"Loaded morphological groups for {len(self.all_phrases)} phrases from cache", file=sys.stderr)


def save_vad():
    print(f"Loading words from vad...", file=sys.stderr)
    words = list(read_words_from_vad())
    print(f"Loaded {len(words)} words/phrases from vad", file=sys.stderr)

    service = MorphologicalGroupingService("vad")
    service.load_from_data(words)
    service.save()


def save_plutchik():
    print(f"Loading words from plutchik...", file=sys.stderr)
    words = list(read_words_from_plutchik())
    print(f"Loaded {len(words)} words/phrases from plutchik", file=sys.stderr)

    service = MorphologicalGroupingService("plutchik")
    service.load_from_data(words)
    service.save()


def load():
    service = MorphologicalGroupingService("vad")
    service.load_from_cache()


if __name__ == '__main__':
    save_vad()
    save_plutchik()
    # load()
