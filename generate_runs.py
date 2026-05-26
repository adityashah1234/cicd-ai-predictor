import os
import time

def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)

def git_push(message):
    os.system("git add .")
    os.system(f'git commit -m "{message}"')
    os.system("git push")
    time.sleep(2)

# ── Clean working versions of all 3 services ─────────────────────────

USER_PASS = '''from flask import Flask, jsonify

app = Flask(__name__)

USERS = [
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    {"id": 2, "name": "Bob",   "email": "bob@example.com"},
    {"id": 3, "name": "Carol", "email": "carol@example.com"},
]

@app.route("/users")
def get_users():
    return jsonify(USERS), 200

@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(port=5001)
'''

ORDER_PASS = '''from flask import Flask, jsonify, request

app = Flask(__name__)
ORDERS = []
counter = 1

@app.route("/orders", methods=["POST"])
def create_order():
    global counter
    d = request.get_json()
    o = {"id": counter, "user_id": d["user_id"], "status": "created"}
    ORDERS.append(o)
    counter += 1
    return jsonify(o), 201

@app.route("/orders")
def get_orders():
    return jsonify(ORDERS), 200

@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(port=5002)
'''

PAYMENT_PASS = '''from flask import Flask, jsonify, request
import uuid

app = Flask(__name__)

@app.route("/pay", methods=["POST"])
def pay():
    d = request.get_json()
    if not d or d.get("amount", 0) <= 0:
        return jsonify({"error": "invalid amount"}), 402
    return jsonify({
        "transaction_id": str(uuid.uuid4()),
        "order_id": d["order_id"],
        "amount": d["amount"],
        "status": "success"
    }), 200

@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(port=5003)
'''

# ── Broken versions that cause test failures ──────────────────────────

USER_FAIL = '''from flask import Flask, jsonify

app = Flask(__name__)

USERS = "this is not a list"

@app.route("/users")
def get_users():
    return jsonify(USERS), 200

@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(port=5001)
'''

ORDER_FAIL = '''from flask import Flask, jsonify, request

app = Flask(__name__)
ORDERS = []
counter = 1

@app.route("/orders", methods=["POST"])
def create_order():
    global counter
    d = request.get_json()
    o = {"id": counter, "user_id": d["user_id"], "status": "created"}
    ORDERS.append(o)
    counter += 1
    return jsonify(o), 200

@app.route("/orders")
def get_orders():
    return jsonify(ORDERS), 200

@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(port=5002)
'''

PAYMENT_FAIL = '''from flask import Flask, jsonify, request
import uuid

app = Flask(__name__)

@app.route("/pay", methods=["POST"])
def pay():
    d = request.get_json()
    if not d or d.get("amount", 0) <= 0:
        return jsonify({"error": "invalid amount"}), 200
    return jsonify({
        "transaction_id": str(uuid.uuid4()),
        "status": "success"
    }), 200

@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(port=5003)
'''

def restore_all():
    write_file("user-service/app.py", USER_PASS)
    write_file("order-service/app.py", ORDER_PASS)
    write_file("payment-service/app.py", PAYMENT_PASS)

# ── PASS patterns ─────────────────────────────────────────────────────
pass_actions = [
    lambda n: (restore_all(), git_push(f"run {n}: pass - user service healthy")),
    lambda n: (restore_all(), git_push(f"run {n}: pass - order service healthy")),
    lambda n: (restore_all(), git_push(f"run {n}: pass - payment service healthy")),
    lambda n: (restore_all(), git_push(f"run {n}: pass - all services clean")),
    lambda n: (restore_all(), git_push(f"run {n}: pass - code review applied")),
]

# ── FAIL patterns ─────────────────────────────────────────────────────
fail_actions = [
    lambda n: (write_file("user-service/app.py", USER_FAIL), git_push(f"run {n}: fail - user-service broken")),
    lambda n: (write_file("order-service/app.py", ORDER_FAIL), git_push(f"run {n}: fail - order-service wrong status")),
    lambda n: (write_file("payment-service/app.py", PAYMENT_FAIL), git_push(f"run {n}: fail - payment-service wrong code")),
]

# ── MAIN ──────────────────────────────────────────────────────────────
print("Starting 100 pipeline run generation...")
print("Leave this running. Takes about 15-20 minutes.")
print("")

run = 1

# 50 PASS runs
print("=== 50 PASS RUNS ===")
for i in range(50):
    action = pass_actions[i % len(pass_actions)]
    action(run)
    print(f"  Pushed run {run}/100 - PASS")
    run += 1

# Always restore before fail runs
restore_all()
git_push("restore: all services working before fail runs")

# 50 FAIL runs - restore after each one
print("\n=== 50 FAIL RUNS ===")
for i in range(50):
    action = fail_actions[i % len(fail_actions)]
    action(run)
    print(f"  Pushed run {run}/100 - FAIL")
    run += 1
    restore_all()
    git_push(f"restore after run {run - 1}")

print("\n" + "=" * 40)
print("ALL 100 RUNS COMPLETE!")
print("Check your GitHub Actions tab.")
print("=" * 40)