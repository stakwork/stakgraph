# @ast node: Endpoint "/lambda-rule"
# @ast node: Function "_lambda_rule_lambda_L14"
# @ast node: Endpoint "/lambda-decorator"
# @ast edge: Handler -> Function "route_lambda_decorator_lambda_L17" "anonymous_functions.py"
# @ast node: Function "route_lambda_decorator_lambda_L17"
# @ast node: Endpoint "/lambda-get"
# @ast edge: Handler -> Function "get_lambda_get_lambda_L20" "anonymous_functions.py"
# @ast node: Function "get_lambda_get_lambda_L20"
# @ast node: Var "app"
from flask import Flask

app = Flask(__name__)

# Test 1: add_url_rule with lambda
app.add_url_rule('/lambda-rule', view_func=lambda: "hello")

# Test 2: route call with lambda (simulated manual decoration)
app.route('/lambda-decorator')(lambda: "world")

# Test 3: get call with lambda (simulated manual decoration)
app.get('/lambda-get')(lambda: "get")
