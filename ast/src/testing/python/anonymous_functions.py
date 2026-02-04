from flask import Flask

app = Flask(__name__)

# Test 1: add_url_rule with lambda
app.add_url_rule('/lambda-rule', view_func=lambda: "hello")

# Test 2: route call with lambda (simulated manual decoration)
app.route('/lambda-decorator')(lambda: "world")

# Test 3: get call with lambda (simulated manual decoration)
app.get('/lambda-get')(lambda: "get")
