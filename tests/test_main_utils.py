from vigil.utils import looks_like_crypto, looks_like_url


def test_looks_like_url():
    assert looks_like_url("https://example.com")
    assert looks_like_url("www.example.com")
    assert not looks_like_url("not-a-url")


def test_looks_like_crypto():
    assert looks_like_crypto("bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kygt080")
    assert looks_like_crypto("0x52908400098527886E0F7030069857D2E4169EE7")
    assert not looks_like_crypto("hello world")
