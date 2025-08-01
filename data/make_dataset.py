# src/data/make_dataset.py
import pandas as pd
from sklearn.datasets import load_iris


def save_raw():
    iris = load_iris(as_frame=True)
    df = pd.concat([iris.data, pd.Series(iris.target, name="target")], axis=1)
    df.to_csv("data/raw/iris.csv", index=False)


if __name__ == "__main__":
    save_raw()
