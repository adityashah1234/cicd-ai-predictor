import pytest
from app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c

def test_successful_payment(client):
    r = client.post("/pay", json={"order_id": 1, "amount": 10.0})
    assert r.status_code == 200
    assert r.get_json()["status"] == "success"

def test_insufficient_funds(client):
    r = client.post("/pay", json={"order_id": 2, "amount": -5.0})
    assert r.status_code == 402

def test_health_check(client):
    r = client.get("/health").get_json()
    assert r["status"] == "healthy"