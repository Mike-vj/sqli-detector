import re

def normalize_query(text: str) -> str:
    """Strip SQL inline comments and normalize whitespace so obfuscated
    payloads like UN/**/ION match their unobfuscated form."""
    text = str(text)
    # Remove /* ... */ style comments (handles UN/**/ION -> UNION)
    text = re.sub(r"/\*.*?\*/", "", text)
    # Collapse multiple spaces/newlines into one
    text = re.sub(r"\s+", " ", text).strip()
    return text