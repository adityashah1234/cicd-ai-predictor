import os
import time

def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)

def git_push(message):
    os.system("git add .")
    result = os.system(f'git commit -m "{message}"')
    if result == 0:
        os.system("git push")
    time.sleep(2)

# Each run writes a slightly different comment so Git sees a real change

def make_user_pass(n):
    return f'''# version {n}
from flask import Flask, jsonify

app = Flask(__name__)

USERS = [
    {{"id": 1, "name": "Alice", "email": "alice@example.com"}},
    {{"id": 2, "name": "Bob",   "email": "bob@example.com"}},
    {{"id": 3, "name": "Carol", "email": "carol@example.com"}},
]

@app.route("/users")
def get_users():
    return jsonify(USERS), 200

@app.route("/health")
def health():
    return jsonify({{"status": "healthy"}}), 200

if __name__ == "__main__":
    app.run(port=5001)
'''

def make_order_pass(n):
    return f'''# version {n}
from flask import Flask, jsonify, request

app = Flask(__name__)
ORDERS = []
counter = 1

@app.route("/orders", methods=["POST"])
def create_order():
    global counter
    d = request.get_json()
    o = {{"id": counter, "user_id": d["user_id"], "status": "created"}}
    ORDERS.append(o)
    counter += 1
    return jsonify(o), 201

@app.route("/orders")
def get_orders():
    return jsonify(ORDERS), 200

@app.route("/health")
def health():
    return jsonify({{"status": "healthy"}}), 200

if __name__ == "__main__":
    app.run(port=5002)
'''

def make_payment_pass(n):
    return f'''# version {n}
from flask import Flask, jsonify, request
import uuid

app = Flask(__name__)

@app.route("/pay", methods=["POST"])
def pay():
    d = request.get_json()
    if not d or d.get("amount", 0) <= 0:
        return jsonify({{"error": "invalid amount"}}), 402
    return jsonify({{
        "transaction_id": str(uuid.uuid4()),
        "order_id": d["order_id"],
        "amount": d["amount"],
        "status": "success"
    }}), 200

@app.route("/health")
def health():
    return jsonify({{"status": "healthy"}}), 200

if __name__ == "__main__":
    app.run(port=5003)
'''

def make_user_fail(n):
    return f'''# version {n}
from flask import Flask, jsonify

app = Flask(__name__)

USERS = "this is broken on purpose"

@app.route("/users")
def get_users():
    return jsonify(USERS), 200

@app.route("/health")
def health():
    return jsonify({{"status": "healthy"}}), 200

if __name__ == "__main__":
    app.run(port=5001)
'''

def make_order_fail(n):
    return f'''# version {n}
from flask import Flask, jsonify, request

app = Flask(__name__)
ORDERS = []
counter = 1

@app.route("/orders", methods=["POST"])
def create_order():
    global counter
    d = request.get_json()
    o = {{"id": counter, "user_id": d["user_id"], "status": "created"}}
    ORDERS.append(o)
    counter += 1
    return jsonify(o), 200

@app.route("/orders")
def get_orders():
    return jsonify(ORDERS), 200

@app.route("/health")
def health():
    return jsonify({{"status": "healthy"}}), 200

if __name__ == "__main__":
    app.run(port=5002)
'''

def make_payment_fail(n):
    return f'''# version {n}
from flask import Flask, jsonify, request
import uuid

app = Flask(__name__)

@app.route("/pay", methods=["POST"])
def pay():
    d = request.get_json()
    if not d or d.get("amount", 0) <= 0:
        return jsonify({{"error": "invalid amount"}}), 200
    return jsonify({{"transaction_id": str(uuid.uuid4()), "status": "success"}}), 200

@app.route("/health")
def health():
    return jsonify({{"status": "healthy"}}), 200

if __name__ == "__main__":
    app.run(port=5003)
'''

pass_makers = [make_user_pass, make_order_pass, make_payment_pass]
pass_files  = ["user-service/app.py", "order-service/app.py", "payment-service/app.py"]

fail_makers = [make_user_fail, make_order_fail, make_payment_fail]
fail_files  = ["user-service/app.py", "order-service/app.py", "payment-service/app.py"]

print("Starting 100 pipeline run generation...")
print("")

run = 1

# 50 PASS runs
print("=== 50 PASS RUNS ===")
for i in range(50):
    idx = i % 3
    write_file(pass_files[idx], pass_makers[idx](run))
    git_push(f"run {run}: pass - {pass_files[idx].split('/')[0]}")
    print(f"  Pushed run {run}/100 - PASS")
    run += 1

# 50 FAIL runs
print("\n=== 50 FAIL RUNS ===")
for i in range(50):
    idx = i % 3
    write_file(fail_files[idx], fail_makers[idx](run))
    git_push(f"run {run}: fail - {fail_files[idx].split('/')[0]}")
    print(f"  Pushed run {run}/100 - FAIL")
    run += 1

# Final restore
print("\nRestoring all services...")
write_file("user-service/app.py", make_user_pass(999))
write_file("order-service/app.py", make_order_pass(999))
write_file("payment-service/app.py", make_payment_pass(999))
git_push("final restore: all services working")

print("\n" + "="*40)
print("ALL 100 RUNS COMPLETE!")
print("Check github.com/adityashah1234/cicd-ai-predictor/actions")
print("="*40)