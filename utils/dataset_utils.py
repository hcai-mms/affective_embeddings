import sys

import numpy as np
import pandas as pd


vad_mapping = {
    'valence': 0, 'arousal': 1, 'dominance': 2
}

def read_words_from_vad(file_path='opendata/NRC-VAD-Lexicon-v2.1.txt'):
    words_vad = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) != 4:
                continue

            word = parts[0]
            try:
                valence = float(parts[1])
                arousal = float(parts[2])
                dominance = float(parts[3])
                words_vad[word] = np.array([valence, arousal, dominance])
            except ValueError:
                continue

    return words_vad

plutchik_mapping = {
    'anger': 0,
    'fear': 1,
    'sadness': 2,
    'joy': 3,
    'disgust': 4,
    'anticipation': 5,
    'trust': 6,
    'surprise': 7
}

def read_words_from_plutchik(file_path='opendata/NRC-Emotion-Intensity-Lexicon-v1.txt'):
    words_emotions = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) != 3:
                continue

            word = parts[0]
            emotion = parts[1]
            try:
                intensity = float(parts[2])
            except ValueError:
                continue

            if word not in words_emotions:
                words_emotions[word] = np.zeros(8)

            if emotion in plutchik_mapping:
                idx = plutchik_mapping[emotion]
                words_emotions[word][idx] = intensity

    return words_emotions


def get_emotional_columns(dataset):
    if dataset == 'vad':
        return list(vad_mapping)
    elif dataset == 'plutchik':
        return list(plutchik_mapping)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'vad' or 'plutchik'")


def get_corpus(vad_path='opendata/NRC-VAD-Lexicon-v2.1.txt',
               plutchik_path='opendata/NRC-Emotion-Intensity-Lexicon-v1.txt',
               goemotion_path='opendata/splits/stratified_splits_5fold_go_emotion.parquet') -> list[str]:
    vad_words = list(read_words_from_vad(vad_path).keys())
    plutchik_words = list(read_words_from_plutchik(plutchik_path).keys())
    goemotion_words = pd.read_parquet(goemotion_path)["sentence_text"].tolist()

    corpus = sorted(set(vad_words + plutchik_words + goemotion_words))
    print(f"Corpus: {len(corpus)} unique items", file=sys.stderr)
    return corpus


if __name__ == '__main__':
    data = read_words_from_vad()
    print()
