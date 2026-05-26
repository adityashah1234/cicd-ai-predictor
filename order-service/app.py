from flask import Flask, jsonify, request

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
