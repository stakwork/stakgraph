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
