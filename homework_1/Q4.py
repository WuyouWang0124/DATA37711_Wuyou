from pathlib import Path

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


def main():
    # Read the data.
    data = pd.read_csv(DATA_PATH)

    X = data[
        ["anchor_ratio", "trans_range", "node_density", "iterations"]
    ]
    y = data["ale"]

    # Use the same split for all models.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Fit standardization using training data only.
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

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    results = []

    for ax, (name, model) in zip(axes, models.items()):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        residuals = y_test.to_numpy() - predictions

        statistic, p_value = stats.normaltest(residuals)

        results.append({
            "Model": name,
            "Test statistic": statistic,
            "p-value": p_value,
            "Decision": (
                "Reject normality"
                if p_value < 0.05
                else "Do not reject normality"
            )
        })

        # Display QQ plots without saving them.
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(f"{name}\np = {p_value:.4f}")
        ax.set_xlabel("Theoretical normal quantiles")
        ax.set_ylabel("Ordered test residuals")
        ax.grid(alpha=0.2)

    # Print the numerical results in the terminal.
    print(f"\nTraining observations: {len(y_train)}")
    print(f"Test observations: {len(y_test)}")
    print("Residual = observed ALE - predicted ALE")
    print("Null hypothesis: residuals follow a normal distribution.")
    print("Significance level: 0.05\n")

    results = pd.DataFrame(results)
    print(
        results.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}"
        )
    )

    print("\nNot rejecting normality does not prove normality.")

    # Open the plots in a window. No output files are created.
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
