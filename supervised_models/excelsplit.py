import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(dataset_path, test_size=0.1):
    df = pd.read_csv(dataset_path)

    fen_list = df['fen'].tolist()
    move_list = df['best_move'].tolist()

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        fen_list, move_list, test_size=test_size, random_state=42
    )

    return X_train, X_test, y_train, y_test
