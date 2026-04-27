
# @ast node: Class "AppConfig"
# @ast edge: Operand -> Function "check_api_key" "config.py"
# @ast edge: Operand -> Function "from_yaml" "config.py"
# @ast node: Function "check_api_key"
# @ast node: Function "from_yaml"
# @ast node: Var "host"
# @ast node: Var "port"
# @ast node: Var "debug"
# @ast node: Var "api_key"
from pydantic import BaseModel, Field, validator
from typing import Optional
import yaml

class AppConfig(BaseModel):
    host: str = "localhost"
    port: int = Field(default=8080, ge=1024, le=65535)
    debug: bool = False
    api_key: Optional[str] = None

    @validator('api_key')
    def check_api_key(cls, v):
        if v and len(v) < 8:
            raise ValueError('API key must be at least 8 characters')
        return v

    @classmethod
    def from_yaml(cls, path: str) -> 'AppConfig':
        # Mock loading
        return cls()
