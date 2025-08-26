import pytest
import json
import io
from pathlib import Path
from server import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'

def test_upload_no_file(client):
    """Test upload without file"""
    response = client.post('/upload')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['error_code'] == 'NO_FILE'

def test_process_page_no_data(client):
    """Test process page without data"""
    response = client.post('/process_page')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['error_code'] == 'NO_DATA'