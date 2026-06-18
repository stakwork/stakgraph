# @ast node: Class "UserRepository"
# @ast edge: Operand -> Function "get" "services.py"
# @ast edge: Operand -> Function "create" "services.py"
class UserRepository:
    # @ast node: Function "get"
    def get(self, user_id: str):
        return {"id": user_id, "name": "Alice"}

    # @ast node: Function "create"
    def create(self, data: dict):
        return {"id": "new-id", **data}


# @ast node: Class "UserService"
# @ast edge: Operand -> Function "get_user" "services.py"
# @ast edge: Operand -> Function "create_user" "services.py"
class UserService:
    # @ast node: Function "__init__"
    def __init__(self):
        self.repo = UserRepository()

    # @ast node: Function "get_user"
    # @ast edge: Calls -> Function "get" "services.py"
    def get_user(self, user_id: str):
        return self.repo.get(user_id)

    # @ast node: Function "create_user"
    # @ast edge: Calls -> Function "create" "services.py"
    def create_user(self, data: dict):
        return self.repo.create(data)


# @ast node: Class "UserController"
# @ast edge: Operand -> Function "handle_get" "services.py"
# @ast edge: Operand -> Function "handle_create" "services.py"
class UserController:
    # @ast node: Function "__init__"
    def __init__(self, service: UserService):
        self.service = service

    # @ast node: Function "handle_get"
    # @ast edge: Calls -> Function "get_user" "services.py"
    def handle_get(self, user_id: str):
        return self.service.get_user(user_id)

    # @ast node: Function "handle_create"
    # @ast edge: Calls -> Function "create_user" "services.py"
    def handle_create(self, data: dict):
        return self.service.create_user(data)
