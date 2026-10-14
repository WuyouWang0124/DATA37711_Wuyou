from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "mcs_ds_edited_iter_shuffled.csv"
RESULTS_DIR = BASE_DIR / "results"


def main():
    # Read the data and select the required variables.
    data = pd.read_csv(DATA_PATH)

    X = data[
        ["anchor_ratio", "trans_range", "node_density", "iterations"]
    ]
    y = data["ale"]

    # Use the same training and test sets for all models.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Scaling is fitted on the training data only.
    models = {
        "Linear regression": make_pipeline(
            StandardScaler(),
            LinearRegression()
        ),
        "SVR (RBF)": make_pipeline(
            StandardScaler(),
            SVR(kernel="rbf", C=1.0, epsilon=0.1)
        ),
        "Random forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42
        )
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    results = []

    for ax, (name, model) in zip(axes, models.items()):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Residual = observed value - predicted value.
        residuals = y_test.to_numpy() - predictions

        # Test the null hypothesis of a normal distribution.
        statistic, p_value = stats.normaltest(residuals)

        results.append({
            "model": name,
            "test_statistic": statistic,
            "p_value": p_value,
            "reject_at_0.05": bool(p_value < 0.05)
        })

        # Draw a normal QQ plot.
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(f"{name}\np = {p_value:.4f}")
        ax.set_xlabel("Theoretical normal quantiles")
        ax.set_ylabel("Ordered test residuals")
        ax.grid(alpha=0.2)

    results = pd.DataFrame(results)
    results.to_csv(RESULTS_DIR / "q4_results.csv", index=False)

    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / "q4_qqplots.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)

    print(f"Training observations: {len(y_train)}")
    print(f"Test observations: {len(y_test)}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
