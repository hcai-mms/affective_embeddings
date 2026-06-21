import io
import os
import sys
import pandas as pd
import requests

from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

EKMAN_MAPPING = {'anger': ['anger', 'annoyance', 'disapproval'],
                 'disgust': ['disgust'],
                 'fear': ['fear', 'nervousness'],
                 'joy': ['joy', 'amusement', 'approval', 'excitement', 'gratitude', 'love', 'optimism', 'relief', 'pride', 'admiration', 'desire', 'caring'],
                 'sadness': ['sadness', 'disappointment', 'embarrassment', 'grief', 'remorse'],
                 'surprise': ['surprise', 'realization', 'confusion', 'curiosity'],
                 'neutral': ['neutral']}

EMOTIONS = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire',
            'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness',
            'surprise', 'neutral']


class GoEmotionService:
    def __init__(self, splits_path="opendata/splits", n_folds=5):
        self.filepath = os.path.join(splits_path, f'stratified_splits_{n_folds}fold_go_emotion.parquet')
        self.df = pd.DataFrame()
        self.n_folds = n_folds

    def load_from_data(self,
                       data_url="https://raw.githubusercontent.com/google-research/google-research/refs/heads/master/goemotions/data"):
        emotions = requests.get(f"{data_url}/emotions.txt").text.splitlines()
        assert EMOTIONS == emotions
        ekman_mapping = requests.get(f"{data_url}/ekman_mapping.json").json()
        ekman_mapping['neutral'] = ['neutral']
        assert EKMAN_MAPPING == ekman_mapping

        inverse_ekman_mapping = {e: k for k, v in ekman_mapping.items() for e in v}

        dfs = []
        for split in ["train", "dev", "test"]:
            r = requests.get(f"{data_url}/{split}.tsv")

            df = pd.read_csv(io.StringIO(r.text),
                             sep='\t',
                             header=None,
                             names=["sentence_text", "category", "id"],
                             index_col="id")

            df["category"] = df["category"].apply(lambda x: [int(e) for e in x.split(',')])
            df["micro_category"] = df["category"].apply(lambda x: [emotions[e] for e in x])
            df["macro_category"] = df["micro_category"].apply(
                lambda x: list({inverse_ekman_mapping[e] for e in x}))
            df["split"] = split
            dfs.append(df)
        self.df = pd.concat(dfs, verify_integrity=True)

    def save(self):
        print(f"Saving {len(self.df)}", file=sys.stderr)
        self.df.to_parquet(self.filepath)
        print("Done!", file=sys.stderr)

    def load_from_cache(self):
       self.df = pd.read_parquet(self.filepath)
       print(f"Loaded {len(self.df)}", file=sys.stderr)

    def stratified(self, seed=42):
        self.df["fold_idx"] = -1
        df_train = self.df[self.df["split"] != "test"]
        df_test = self.df[self.df["split"] == "test"]

        y_list = df_train["category"].tolist()

        mlb = MultiLabelBinarizer()
        Y = mlb.fit_transform(y_list)

        mskf = MultilabelStratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=seed)

        for fold, (_, val_idx) in enumerate(mskf.split(df_train.index.to_numpy(), Y)):
            df_train.iloc[val_idx, df_train.columns.get_loc("fold_idx")] = fold

        self.df = pd.concat([df_train, df_test])


def download():
    go_emotion_service = GoEmotionService()
    go_emotion_service.load_from_data()
    go_emotion_service.save()


def stratify():
    go_emotion_service = GoEmotionService()
    go_emotion_service.load_from_cache()
    go_emotion_service.stratified()
    go_emotion_service.save()
    print('Finished')

def load():
    go_emotion_service = GoEmotionService()
    go_emotion_service.load_from_cache()
    print('Finished')


if __name__ == '__main__':
    # download()
    # stratify()
    load()
