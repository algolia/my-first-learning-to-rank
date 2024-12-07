import os
from pathlib import Path
import pandas as pd


def load_dataset(sample: int = 0, *, local: bool = False) -> pd.DataFrame:
    if local:
        here = Path(__file__)
        source = here.parent.parent.parent / "data/dataset.csv"
    else:
        source = "https://raw.githubusercontent.com/algolia/my-first-learning-to-rank/main/data/dataset.csv"

    if sample:
        return pd.read_csv(source).sample(sample)
    return pd.read_csv(source)
