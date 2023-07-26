import os

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
    assert response.json()["data"][0][0]["label"] == "Healthy"
    assert response.json()["data"][0][0]["score"] > 0.9


def test_classify_early_blight(app_client):
    response = app_client.post(
        "/classify",
        files={"file": open("./tests/images/early_blight.jpeg", "rb")},
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Successful classification"
    assert response.json()["data"][0][0]["label"] == "Early_Blight"
    assert response.json()["data"][0][0]["score"] > 0.9


def test_classify_late_blight(app_client):
    response = app_client.post(
        "/classify",
        files={"file": open("./tests/images/late_blight.jpeg", "rb")},
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Successful classification"
    assert response.json()["data"][0][0]["label"] == "Late_Blight"
    assert response.json()["data"][0][0]["score"] > 0.9

def test_bad_validation_images(app_client):
    path = r"E:/dev/ML-take-home/ml-client/public/images/PLD_3_Classes_256/test"

    dirty = []

    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if ".jpg" in name:
                response = app_client.post(
                    "/classify",
                    files={"file": open(os.path.join(root, name), "rb")},
                )

                if response.json()["data"][0][0]["label"] not in name:
                    dirty.append(name)

        print(dirty)

        assert len(dirty) == 0
                
