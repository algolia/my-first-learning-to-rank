import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import download

# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer

# Download necessary NLTK data
download("punkt")
download("punkt_tab")
download("stopwords")


# FIXME: normalize query, attribute is query as you type
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
    # Remove stopwords
    # stop_words = set(stopwords.words('english'))
    # tokens = [word for word in tokens if word not in stop_words]
    # Stemming
    # stemmer = PorterStemmer()
    # tokens = [stemmer.stem(word) for word in tokens]
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
