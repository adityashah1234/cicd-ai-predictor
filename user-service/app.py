# version 72
from flask import Flask, jsonify

app = Flask(__name__)

USERS = "this is broken on purpose"

@app.route("/users")
def get_users():
    return jsonify(USERS), 200

@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(port=5001)
