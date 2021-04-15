import os
import pandas as pd

from datasets.download import HOUSING_PATH, fetch_housing_data


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    if not os.path.isfile(csv_path):
        fetch_housing_data()
    return pd.read_csv(csv_path)


if __name__ == '__main__':
    housing_data = load_housing_data()
    print(housing_data)