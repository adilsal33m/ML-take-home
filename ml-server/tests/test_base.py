def test_base(app_client):
    response = app_client.get("/")
    assert response.status_code == 200
    assert response.json()["message"].startswith("Welcome to the image classifier")


def test_classify_healthy(app_client):
    response = app_client.post(
        "/classify",
        files={"file": open("./tests/images/healthy.jpeg", "rb")},
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Successful classification"
    assert response.json()["data"][0]["label"] == "Healthy"
    assert response.json()["data"][0]["score"] > 0.9


def test_classify_early_blight(app_client):
    response = app_client.post(
        "/classify",
        files={"file": open("./tests/images/early_blight.jpeg", "rb")},
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Successful classification"
    assert response.json()["data"][0]["label"] == "Early_Blight"
    assert response.json()["data"][0]["score"] > 0.9


def test_classify_late_blight(app_client):
    response = app_client.post(
        "/classify",
        files={"file": open("./tests/images/late_blight.jpeg", "rb")},
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Successful classification"
    assert response.json()["data"][0]["label"] == "Late_Blight"
    assert response.json()["data"][0]["score"] > 0.9
