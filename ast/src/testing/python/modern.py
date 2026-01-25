
from dataclasses import dataclass
import asyncio

# Typed assignment (global)
typed_global: int = 42

@dataclass
class User:
    name: str
    age: int

    def get_info(self) -> str:
        return f"{self.name} is {self.age}"

async def fetch_data(url: str):
    # Async function definition
    print(f"Fetching {url}")
    await asyncio.sleep(1)
    return {"data": "ok"}

class AsyncProcessor:
    @staticmethod
    async def process(item):
        # Async method in class
        return item * 2
