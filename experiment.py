# numpy
# xlrd
# matplotlib
# statsmodels
# scikit-learn


import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import xlrd
from sklearn.linear_model import LinearRegression

from gradient_descent import gradient_descent

def metrics(y, prediction):
    mse = float(np.mean((prediction - y) ** 2))
    return mse, 1 - mse / float(np.var(y))


def table(output, name, headers, rows, picture=False):
    with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
    cells = [[f"{v:.6g}" if isinstance(v, (float, np.floating)) else str(v)
              for v in row] for row in rows]
    if picture:
        fig, ax = plt.subplots(figsize=(12, 1.3 + 0.45 * len(rows)))
        ax.axis("off")
        display = ax.table(cellText=cells, colLabels=headers, loc="center", cellLoc="center")
        display.auto_set_font_size(False)
        display.set_fontsize(9)
        display.scale(1, 1.7)
        display.auto_set_column_width(list(range(len(headers))))
        fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main():
    root = Path(__file__).resolve().parent
    iterations = 100000
    sheet = xlrd.open_workbook(str(root / "Concrete_Data.xls")).sheet_by_index(0)
    data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)], dtype=float)
    features = ["Cement", "Blast furnace slag", "Fly ash", "Water", "Superplasticizer",
                "Coarse aggregate", "Fine aggregate", "Age"]
    mask = np.ones(1030, dtype=bool)
    mask[500:630] = False
    x, y = data[mask, :8], data[mask, 8]
    xt, yt = data[~mask, :8], data[~mask, 8]
    mean, scale = x.mean(axis=0), x.std(axis=0)

    configs = {"Untransformed": (x, xt),
               "Standardized": ((x - mean) / scale, (xt - mean) / scale),
               "Log-Transformed": (np.log1p(x), np.log1p(xt))}
    configs.update({name: (x[:, j:j+1], xt[:, j:j+1]) for j, name in enumerate(features)})
    
    fits, histories, training, optimal = {}, {}, [], []
    for name, (a, at) in configs.items():
        design = sm.add_constant(a, has_constant="add")
        test_design = sm.add_constant(at, has_constant="add")

        # change learning rate here
        
        learning_rate = 0.9 / np.linalg.eigvalsh(2 * design.T @ design / len(y))[-1]
        theta, loss = gradient_descent(a, y, learning_rate, iterations)
        histories[name] = loss
        train_mse, train_r2 = metrics(y, design @ theta)
        test_mse, test_r2 = metrics(yt, test_design @ theta)
        positive = np.flatnonzero(loss < np.var(y))
        training.append([name, learning_rate, int(positive[0]) if len(positive) else "not reached",
                         train_mse, test_mse, train_r2, test_r2])
        fit = LinearRegression().fit(a, y)
        fits[name] = fit
        train_mse, train_r2 = metrics(y, fit.predict(a))
        test_mse, test_r2 = metrics(yt, fit.predict(at))
        optimal.append([name, train_mse, test_mse, train_r2, test_r2])


    output = root / "results"
    output.mkdir(exist_ok=True)
    model_names = list(configs)[:3]
    raw = fits["Untransformed"]
    coefficients = [[name, fits[name].coef_[0], raw.coef_[j]] for j, name in enumerate(features)]
    statistics = {name: sm.OLS(y, sm.add_constant(configs[name][0], has_constant="add")).fit()
                  for name in model_names}
    pvalues = [[name] + [statistics[model].pvalues[j+1] for model in model_names] for j, name in enumerate(features)]
    table(output, "training", ["Model", "Learning rate", "First positive R²", "Train MSE", "Test MSE", "Train R²", "Test R²"], training)
    metric_headers = ["Model", "Training MSE", "Testing MSE", "Training R²", "Testing R²"]
    table(output, "optimal_metrics", metric_headers, optimal[:3], True)
    table(output, "univariate_metrics", metric_headers, optimal[3:])
    table(output, "coefficients", ["Feature", "Univariate slope", "Multivariate slope"], coefficients, True)
    table(output, "pvalues", ["Feature", "Untransformed", "Standardized", "Log-Transformed"], pvalues, True)
    for count, filename in [(2, "q3_loss.png"), (3, "q5_loss.png")]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for name in model_names[:count]:
            ax.plot(np.arange(iterations + 1), histories[name], label=name)
        ax.set_xscale("symlog", linthresh=10)
        ax.set(xlabel="Iterations (symlog scale)", ylabel="Training MSE (MPa squared)")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)

if __name__ == "__main__":
    main()
