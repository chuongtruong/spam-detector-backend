from app import preprocess
def test_change():
    assert preprocess('hello world') is not None