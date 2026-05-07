from data_ingest import hello


def test_hello() -> None:
    assert hello() == "Hello from data-ingest!"
