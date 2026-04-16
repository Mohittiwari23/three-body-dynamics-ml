import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from three_body.visualize3 import animate_3body


def visualize_row(csv_path, row_idx):

    df = pd.read_csv(csv_path)
    row = df.iloc[row_idx]

    traj_path = row["traj_file"]

    data = np.load(traj_path)

    class Result:
        pass

    result = Result()
    result.traj = data["traj"]
    result.time = data["time"]
    result.E_hist = data["E_hist"]
    result.L_hist = data["L_hist"]

    # FIX: derive instead of loading
    result.E0 = result.E_hist[0]
    result.L0 = result.L_hist[0]

    result.MEGNO_final = row["MEGNO"]

    print("\n=== ROW INFO ===")
    print(row[["regime", "outcome", "MEGNO", "dE_max"]])

    animate_3body(result)


if __name__ == "__main__":
    visualize_row(
        csv_path="data/dataset3_body/gen_data-200.csv",
        row_idx=98
    )