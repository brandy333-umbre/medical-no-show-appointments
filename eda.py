import pandas as pd

DATA_PATH = "data/noshow.csv"

def main():
    df = pd.read_csv(DATA_PATH)

    print("Shape:", df.shape)
    print("\nColumns:\n", df.columns)

    print("\nTarget distribution (counts):")
    print(df["No-show"].value_counts())

    print("\nTarget distribution (percentage):")
    print(df["No-show"].value_counts(normalize=True))

    print("\nMissing values:")
    print(df.isnull().sum())

if __name__ == "__main__":
    main()
