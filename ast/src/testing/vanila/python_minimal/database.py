from typing import Dict, Any, List, Optional
from .config import settings

class Database:
    def __init__(self):
        self.url = settings.DATABASE_URL
        self._store: Dict[int, Dict[str, Any]] = {}
        self._id_counter = 1

    def connect(self):
        print(f"Connecting to database at {self.url}...")
        return True

    def insert(self, table: str, data: Dict[str, Any]) -> int:
        # Mocking table logic by just dumping into the same store for simplicity
        record_id = self._id_counter
        data['id'] = record_id
        self._store[record_id] = data
        self._id_counter += 1
        return record_id

    def get(self, table: str, record_id: int) -> Optional[Dict[str, Any]]:
        return self._store.get(record_id)

    def all(self, table: str) -> List[Dict[str, Any]]:
        return list(self._store.values())

db = Database()
