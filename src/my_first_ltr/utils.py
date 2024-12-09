import json
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


def load_raw_dataset(local: bool = False) -> pd.DataFrame:
    """Load dataset with categories not hot encoded for data exploration."""
    if local:
        here = Path(__file__)
        source = here.parent.parent.parent / "data/dataset_with_cats.csv"
    else:
        source = "https://raw.githubusercontent.com/algolia/my-first-learning-to-rank/main/data/dataset_with_cats.csv"

    dataset = pd.read_csv(
        source,
        converters={"genres": json.loads, "production_countries": json.loads},
    )
    return dataset
