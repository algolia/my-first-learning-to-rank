import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import download

# Download necessary NLTK data
download("punkt")
download("punkt_tab")
download("stopwords")


def normalize(text: str) -> str:
    # Convert to lowercase
    text = text.lower()
    # Remove non-alphanumeric characters
    text = re.sub(r"\W+", " ", text)
    text = text.strip()
    return text


def tokenize(text: str) -> list[str]:
    # Tokenize text
    tokens = word_tokenize(text)
    return tokens


# Normalization and tokenization function
def normalize_and_tokenize(text: str) -> list[str]:
    text = normalize(text)
    tokens = tokenize(text)
    return tokens


def normalize_field(df: pd.DataFrame, field: str, new_field: str) -> pd.DataFrame:
    df.dropna(subset=[field], inplace=True)
    df[f"normalized_{new_field}"] = df[field].apply(normalize)
    df.reset_index(inplace=True, drop=True)
    return df
