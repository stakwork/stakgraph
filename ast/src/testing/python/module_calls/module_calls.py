# @ast node: Class "Record"
# @ast edge: Operand -> Function "process" "module_calls.py"
class Record:
    # @ast node: Function "process"
    def process(self) -> str:
        return "done"


# @ast node: Class "Store"
# @ast edge: Operand -> Function "get_record" "module_calls.py"
class Store:
    # @ast node: Function "get_record"
    def get_record(self) -> Record:
        return Record()


# @ast node: Function "run_method_chain"
# @ast edge: Calls -> Function "get_record" "module_calls.py"
# @ast edge: Calls -> Function "process" "module_calls.py"
def run_method_chain() -> str:
    store = Store()
    record = store.get_record()
    return record.process()


# @ast node: Function "fetch_data"
def fetch_data(url: str) -> dict:
    return {"url": url}


# @ast node: Function "transform"
def transform(data: dict) -> dict:
    return {k: str(v) for k, v in data.items()}


# @ast node: Function "pipeline"
# @ast edge: Calls -> Function "fetch_data" "module_calls.py"
# @ast edge: Calls -> Function "transform" "module_calls.py"
def pipeline(url: str) -> dict:
    raw = fetch_data(url)
    return transform(raw)
