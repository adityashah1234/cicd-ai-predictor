import pytest
from app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c

def test_create_order_returns_201(client):
    r = client.post("/orders", json={"user_id": 1, "items": []})
    assert r.status_code == 201
    assert r.get_json()["status"] == "created"

def test_get_orders_returns_200(client):
    assert client.get("/orders").status_code == 200

def test_health_check(client):
    r = client.get("/health").get_json()
    assert r["status"] == "healthy"