import pandas as pd

def load_data():
    df = pd.read_csv(r'C:\Users\Akshara\PycharmProjects\PlacementPredictionSystem\data\placement_data.csv')
    return df

def get_summary(df):
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "target": "Placement status"
    }

if __name__ == "__main__":
    df = load_data()

    print("Dataset Summary:")
    print(get_summary(df))

    print("\nFirst 5 Rows of the Dataset:")
    print(df.head())