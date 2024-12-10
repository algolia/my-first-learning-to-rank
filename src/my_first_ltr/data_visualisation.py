import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoost, Pool, EFstrType


RMSE: str = "RMSE"
MAP: str = "MAP"
MAE: str = "MAE"
NDCG: str = "NDCG:top=-1;type=Base;denominator=LogPosition"


def plot_eval_results(model_name: str, model: CatBoost) -> None:
    """Plot metrics per iterations during training."""
    eval_results = model.get_evals_result()

    for k, metrics in list(eval_results.items()):
        for m, y in list(metrics.items()):
            plt.plot(y, label=k + "-" + m)
    plt.legend()
    # plt.xlabel("Iterations (/200)")
    # plt.ylabel("Loss")
    plt.legend()
    plt.title(model_name)
    plt.show()


def _get_flat_shap_values(m: CatBoost, pool: Pool) -> list:
    # Get SHAP values
    shap_values = m.get_feature_importance(pool, EFstrType.ShapValues)

    # Separate SHAP values and predictions
    shap_values_only = shap_values[:, :-1]  # Exclude the last column (model output)
    shap_mean_abs = np.mean(np.abs(shap_values_only), axis=0)
    return shap_mean_abs


def cross_dataset_plot_shap_values(
    m: CatBoost, pools: dict[str, Pool], top_n: int = 20
) -> None:
    # Create a DataFrame for plotting
    features = pools["train"].get_feature_names()
    shap_per_dataset = (
        pd.DataFrame(
            data={k: _get_flat_shap_values(m, pool) for k, pool in pools.items()},
            index=features,
        )
        .sort_values(by="train", ascending=False)
        .head(top_n)
    )

    # Plot the results
    plt.figure(figsize=(10, 6))
    shap_per_dataset.plot.barh(figsize=(10, 6))
    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Feature")
    plt.title("Feature Importance based on SHAP Values")
    plt.gca().invert_yaxis()  # Invert y-axis for descending order
    plt.show()
