import pytest
from app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c

def test_get_users_returns_200(client):
    assert client.get("/users").status_code == 200

def test_users_have_required_fields(client):
    users = client.get("/users").get_json()
    assert len(users) > 0
    assert "id" in users[0]
    assert "name" in users[0]

def test_health_check(client):
    r = client.get("/health").get_json()
    assert r["status"] == "healthy"