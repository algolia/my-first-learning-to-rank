import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import Pool


def get_categories(dataset: pd.DataFrame) -> list[str]:
    categories = dataset.select_dtypes(["object", "category"]).columns.to_list()
    return categories


def _group_per_query(dataset: pd.DataFrame) -> pd.DataFrame:
    # Define all features we need to keep for the pool
    pool_features = list(set(dataset.columns) - set(["normalized_query"]))

    # group per query and map the records data
    #   query                                                      hits
    # 0     a  [{'brand': '1', 'price': 1}, {'brand': '2', 'price': 2}]
    grouped_per_query = (
        dataset.groupby(
            ["normalized_query"], sort=False, group_keys=True, as_index=True
        )[pool_features]
        .apply(lambda x: x.to_dict("records"))
        .reset_index()
        .rename(
            columns={0: "data"}
        )  # FIXME I think drop=True) in reset index does the same
    )

    return grouped_per_query


def _explode_per_query(grouped_dataset: pd.DataFrame) -> pd.DataFrame:
    """Inverse operation of group per query"""

    # Recreate one row per query / data
    dataset = grouped_dataset.explode("data")

    # remap the data content to columns
    dataset = pd.concat([dataset, dataset["data"].apply(pd.Series)], axis=1).drop(
        columns=["data"]
    )

    return dataset


def dataset_split(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped_per_query = _group_per_query(dataset)

    train_test_queries_df, val_queries_df = train_test_split(
        grouped_per_query, test_size=0.05, shuffle=False
    )
    train_queries_df, test_queries_df = train_test_split(
        train_test_queries_df, test_size=0.3, shuffle=False
    )

    train_df = _explode_per_query(train_queries_df)
    test_df = _explode_per_query(test_queries_df)
    val_df = _explode_per_query(val_queries_df)

    print("After split: train dataset", train_df.shape)
    print("After split: test dataset", test_df.shape)
    print("After split: validation dataset", val_df.shape)

    return train_df, test_df, val_df


def validate_not_empty(df: pd.DataFrame, name: str) -> None:
    """Ensure a dataframe is not empty. Raise an error if it's the case."""
    if df.shape[0] == 0:
        raise ValueError(f"Dataset {name} should not be empty, shape={df.shape}")


def keep_input_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Filter out target, queries and item id."""
    return dataset.drop(columns=["normalized_query", "id", "score"])


def build_pool(dataset: pd.DataFrame, name: str) -> Pool:
    """Given the config and the dataset build the Catboost pool"""
    validate_not_empty(dataset, name)
    dataset = dataset.sort_values(by=["normalized_query"])

    X = keep_input_features(dataset)
    y = dataset["score"]
    queries = dataset["normalized_query"].values

    categories = get_categories(X)

    pool = Pool(
        data=X,
        label=y,
        group_id=queries,
        cat_features=categories,
        thread_count=1,
    )
    return pool
