import re

EMAIL_REGEX = re.compile(r"[^@]+@[^@]+\.[^@]+")

def is_valid_email(email: str) -> bool:
    return bool(EMAIL_REGEX.match(email))

def is_strong_password(password: str) -> bool:
    return len(password) >= 8
