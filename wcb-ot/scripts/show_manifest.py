import pandas as pd

def show_results():
    df = pd.read_csv("data/processed/manifest.csv")
    print("Manifest Head:")
    print(df.head())
    print("\nClass Counts:")
    print(df.groupby('class').size())
    print("\nNC Ratio Statistics:")
    print(df['nc_ratio'].describe())
    print(f"\nTotal images: {len(df)}")

if __name__ == "__main__":
    show_results()
