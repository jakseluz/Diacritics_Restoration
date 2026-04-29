from datasets import load_dataset
import pandas as pd
import random

import torch
from torch.utils.data import Dataset, DataLoader

from .processor import CharacterProcessor


class DiacriticsDataset(Dataset):
    def __init__(self, texts: list[str], window_size: int = 256, step: int = 100):
        self.window_size = window_size
        self.processor = CharacterProcessor()

        self.linux_x = []  # :)
        self.linux_y = []

        mapping = {
            "ą": "a",
            "ć": "c",
            "ę": "e",
            "ł": "l",
            "ń": "n",
            "ó": "o",
            "ś": "s",
            "ź": "z",
            "ż": "z",
            "Ą": "A",
            "Ć": "C",
            "Ę": "E",
            "Ł": "L",
            "Ń": "N",
            "Ó": "O",
            "Ś": "S",
            "Ź": "Z",
            "Ż": "Z",
        }
        self.trans_table = str.maketrans(mapping)

        for text in texts:
            for i in range(0, len(text) - window_size + 1, step):
                y_window = text[i : i + window_size]
                x_window = y_window.translate(self.trans_table)

                self.linux_x.append(x_window)
                self.linux_y.append(y_window)

    def __len__(self):
        return len(self.linux_x)

    def __getitem__(self, idx):
        x_ids = self.processor.text_to_sequence(self.linux_x[idx])
        y_ids = self.processor.text_to_sequence(self.linux_y[idx])

        if len(y_ids) < self.window_size:
            padding_len = self.window_size - len(y_ids)
            y_ids += [self.processor.pad_token_id] * padding_len
            x_ids += [self.processor.pad_token_id] * padding_len

        x_ids = x_ids[: self.window_size]
        y_ids = y_ids[: self.window_size]

        return torch.tensor(x_ids, dtype=torch.long), torch.tensor(y_ids, dtype=torch.long)


def get_wikipedia_data(num_articles=100, wiki_version="20231101.pl") -> pd.DataFrame:
    """
    Load a sample of articles from the Polish Wikipedia dataset.
    Args:
        num_articles (int): The number of articles to sample from the Wikipedia dataset.
        wiki_version (str): The version of the Wikipedia dataset to load.
    Returns:
        pd.DataFrame: A DataFrame containing the sampled Wikipedia articles.
    """
    df_wiki = None
    try:
        wiki = load_dataset("wikimedia/wikipedia", wiki_version, split="train")
        print("Polish Wikipedia successfully loaded!")
        wiki.set_format("pandas")
        df_wiki = wiki[:]
        print("Wikipedia dataset converted to DataFrame!")
    except Exception as e:
        print(f"Error loading Wikipedia: {e}")

    indexes = random.sample(range(len(df_wiki)), num_articles)
    df_wiki: pd.DataFrame = df_wiki.iloc[indexes].reset_index(drop=True)
    return df_wiki
