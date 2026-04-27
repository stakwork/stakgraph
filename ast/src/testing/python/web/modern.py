# @ast node: Var "typed_global"
# @ast node: Class "User"
# @ast edge: Operand -> Function "get_info" "modern.py"
# @ast node: Var "name"
# @ast node: Var "age"
# @ast node: Function "get_info"
# @ast node: Function "fetch_data"
# @ast node: Class "AsyncProcessor"
# @ast edge: Operand -> Function "process" "modern.py"
# @ast node: Function "process"
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
