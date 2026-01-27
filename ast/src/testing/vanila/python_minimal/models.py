from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class User:
    id: Optional[int]
    username: str
    email: str
    created_at: datetime = datetime.now()

@dataclass
class Post:
    id: Optional[int]
    user_id: int
    content: str
    published: bool = False
