import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import json
from databases import Database
import asyncio
from typing import Generator
import os
import aiosqlite


from ..dms import app, Rule, Document, TrainingData


client = TestClient(app)

TEST_DATABASE_URL = "sqlite:///test_rules.db"

@pytest.fixture(scope="session")
def event_loop() -> Generator:
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
async def test_db():
    if os.path.exists("test_rules.db"):
        os.remove("test_rules.db")
    
    async with aiosqlite.connect("test_rules.db") as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT,
                category TEXT,
                components TEXT,
                rule_type TEXT,
                priority INTEGER,
                confidence FLOAT,
                created_at TIMESTAMP
            )
        """)
        await db.commit()
    
    yield
    
    if os.path.exists("test_rules.db"):
        os.remove("test_rules.db")

@pytest.fixture
def sample_training_data():
    return {
        "examples": [
            {
                "text": "باید گزارش ماهانه را تا تاریخ 5 ام ماه ارسال کنید",
                "is_rule": True
            },
            {
                "text": "جلسه امروز ساعت 14 برگزار می‌شود",
                "is_rule": False
            }
        ]
    }

@pytest.fixture
def sample_document():
    return {
        "content": "گزارش ماهانه شامل آمار فروش و هزینه‌ها می‌باشد.",
        "title": "گزارش ماهانه",
        "type": "report"
    }

@pytest.fixture
def sample_email():
    return "باید گزارش‌های هفتگی تا روز پنجشنبه ارسال شود"


def test_api_root():
    """Test the API root endpoint."""
    response = client.get("/")
    assert response.status_code == 404

def test_openapi_docs():
    """Test that the API documentation is accessible."""
    response = client.get("/docs")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_train_system_success(sample_training_data):
    """Test successful training of the system."""
    response = client.post(
        "/api/train",
        json=sample_training_data
    )
    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "message": "System trained successfully"
    }

@pytest.mark.asyncio
async def test_train_system_invalid_data():
    """Test training with invalid data."""
    response = client.post(
        "/api/train",
        json={"examples": []}
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_extract_rules(sample_training_data, sample_email):
    train_response = client.post(
        "/api/train",
        json=sample_training_data
    )
    assert train_response.status_code == 200

    response = client.post(
        "/api/rules/extract",
        params={"email_content": sample_email}
    )
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "rules" in data
    assert isinstance(data["rules"], list)

@pytest.mark.asyncio
async def test_extract_rules_empty_content():
    """Test rule extraction with empty content."""
    response = client.post(
        "/api/rules/extract",
        params={"email_content": ""}
    )
    assert response.status_code == 200
    assert response.json()["rules"] == []


@pytest.mark.asyncio
async def test_validate_document(sample_document):
    async with aiosqlite.connect("test_rules.db") as db:
        await db.execute(
            """
            INSERT INTO rules (content, category, components, rule_type, priority, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "باید گزارش ماهانه ارسال شود",
                "deadline",
                json.dumps({"subject": "گزارش", "deadline": "ماهانه"}),
                "obligation",
                1,
                0.9,
                str(datetime.now())
            )
        )
        await db.commit()

    response = client.post(
        "/api/documents/validate",
        json=sample_document
    )
    assert response.status_code == 200
    
    data = response.json()
    assert "is_valid" in data
    assert "discrepancies" in data
    assert "recommendations" in data

@pytest.mark.asyncio
async def test_validate_invalid_document():
    response = client.post(
        "/api/documents/validate",
        json={"invalid": "document"}
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_full_workflow(sample_training_data, sample_email, sample_document):
    train_response = client.post(
        "/api/train",
        json=sample_training_data
    )
    assert train_response.status_code == 200

    extract_response = client.post(
        "/api/rules/extract",
        params={"email_content": sample_email}
    )
    assert extract_response.status_code == 200

    validate_response = client.post(
        "/api/documents/validate",
        json=sample_document
    )
    assert validate_response.status_code == 200


@pytest.mark.asyncio
async def test_nonexistent_endpoint():
    response = client.post("/api/nonexistent")
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_invalid_json():
    response = client.post(
        "/api/train",
        data="invalid json"
    )
    assert response.status_code == 422

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )

async def cleanup_database():
    if os.path.exists("test_rules.db"):
        os.remove("test_rules.db")
