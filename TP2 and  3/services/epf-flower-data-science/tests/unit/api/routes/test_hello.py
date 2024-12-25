import pytest
from fastapi.testclient import TestClient
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..', '..')))
from main import get_application

class TestHelloRoute:
    @pytest.fixture
    def client(self) -> TestClient:
        """
        Test client for integration tests
        """
        app = get_application()

        client = TestClient(app, base_url="http://testserver")

        return client

    def test_hello(self, client):
        # Setup some test data
        name = "testuser"
        url = f"/hello/{name}"

        # Call the function to be tested
        response = client.get(url)

        # Assert the output
        assert response.status_code == 200
        assert response.json() == {
            "message": "Hello testuser, from fastapi test route !"
        }
