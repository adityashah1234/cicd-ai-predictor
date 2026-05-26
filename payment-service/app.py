# version 39
from flask import Flask, jsonify, request
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
