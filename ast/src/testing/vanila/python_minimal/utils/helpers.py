import random
import string

def generate_token(length: int = 16) -> str:
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

def slugify(text: str) -> str:
    return text.lower().replace(" ", "-")
