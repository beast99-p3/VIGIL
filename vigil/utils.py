import re


def looks_like_crypto(data: str) -> bool:
    if data.startswith("bc1") or data.startswith("1") or data.startswith("3"):
        return True
    if re.fullmatch(r"[13][a-km-zA-HJ-NP-Z1-9]{25,34}", data):
        return True
    if re.fullmatch(r"0x[a-fA-F0-9]{40}", data):
        return True
    return False


def looks_like_url(data: str) -> bool:
    return bool(re.match(r"^(https?://|www\.)", (data or "").lower()))
