from datasets import load_dataset
import pandas as pd
import random


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


def non_polish_cleanup(text: str) -> str:
    """
    Clean up the text by removing non-Polish characters.
    Args:
        text (str): The input text to clean.
    Returns:
        str: The cleaned text.
    """
    characters = set(text)
    polish_characters = set("aąbcćdeęfghijklłmnńoópqrsśtuvwxyzźżAĄBCĆDEĘFGHIJKLŁMNŃOÓPQRSŚTUVWXYZŹŻ")

    to_remove = "".join([char for char in characters if char.isalpha() and char not in polish_characters])

    table = str.maketrans("", "", to_remove)
    return text.translate(table)
