from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class Item(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    created_at: datetime = datetime.now()
    tags: List[str] = []

    def summary(self) -> str:
        return f"{self.name} ({len(self.tags)} tags)"

class ProcessResult(BaseModel):
    success: bool
    processed_count: int
    errors: List[str] = []
