import pandas as pd
import numpy as np


def test_line_segment():
    df = pd.read_excel(".\\Rental_geocode\\policy1_segments.xlsx")

    R = 6371

    df["X"] = np.radians(df["_X"])
    df["Y"] = np.radians(df["_Y"])
    df["next_X"] = df["X"].shift(-1)
    df["next_Y"] = df["Y"].shift(-1)

    df["next_X"].iloc[-1] = df["X"].iloc[0]
    df["next_Y"].iloc[-1] = df["Y"].iloc[0]


    df["dlon"] = df["next_X"] - df["X"]
    df["dlat"] = df["next_Y"] - df["Y"]
    df["a"] = (np.sin(df["dlat"]/2) ** 2) + np.cos(df["X"]) * np.cos(df["next_X"]) * (np.sin(df["dlon"]/2) ** 2)

    df["distance_n"] = 2 * np.arcsin(np.sqrt(df["a"])) * R * 1000
    df["acul_dist_n"] = df["distance_n"].cumsum()

    df = df.drop(columns=["X", "Y", "next_X", "next_Y", "dlon", "dlat", "a"])

    print(df)


def no_return():
    print(1+1)


def subset_check():
    a = [1,2,3,4,5]
    b = {1,2}

    print(b.issubset(a))


def main():
    # test_line_segment()
    no_return()
    


if __name__ == "__main__":
    main()