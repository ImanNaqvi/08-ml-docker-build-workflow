from src.preprocess import load_and_clean

def test_preprocess_not_empty():
    df = load_and_clean("data/sample.csv")
    assert not df.empty

def test_preprocess_no_missing_values():
    df = load_and_clean("data/sample.csv")
    assert df.isna().sum().sum() == 0
