from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

SEED = 42


def load_spambase():
    """Spambase (UCI) from OpenML: 4,601 emails x 57 numeric features, binary spam label."""
    data = fetch_openml("spambase", version=1, as_frame=True, parser="auto")
    X = data.data.astype("float64")
    y = data.target.astype(int)  # labels load as the strings '0'/'1'
    return X, y


def load_splits():
    """Fixed stratified 80/20 split, shared by training and evaluation so the
    evaluation set is never seen during training."""
    X, y = load_spambase()
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
