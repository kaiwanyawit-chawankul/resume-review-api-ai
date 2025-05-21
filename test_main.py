# test_main.py
from fastapi.testclient import TestClient
from main import app, ResumeReviewResponse
import pytest
from unittest.mock import patch
import json

# Create a TestClient instance for the FastAPI app
client = TestClient(app)

@pytest.fixture
def mock_gemini_response():
    """
    Fixture to provide a mocked Gemini API response.
    """
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps({
                                "review": "The resume shows good potential but needs tailoring.",
                                "strengths": ["Strong technical skills", "Relevant project experience"],
                                "weaknesses": ["Lack of specific achievements", "Generic summary"],
                                "suggestions": ["Quantify achievements", "Tailor summary to job description"],
                                "matched_keywords": ["Python", "FastAPI", "Docker"],
                                "missing_keywords": ["Kubernetes", "CI/CD"]
                            })
                        }
                    ]
                }
            }
        ]
    }

@patch('google.generativeai.GenerativeModel')
def test_review_resume_success(mock_generative_model, mock_gemini_response):
    """
    Test the /review_resume endpoint for a successful response.
    Mocks the Gemini API call to return a predefined response.
    """
    # Configure the mock object to return the predefined response
    mock_instance = mock_generative_model.return_value
    mock_instance.generate_content_async.return_value = mock_gemini_response

    # Define the request payload
    request_payload = {
        "job_description": "We are looking for a Python developer with FastAPI and Docker experience.",
        "resume_text": "Experienced developer with Python and Docker skills. Built APIs."
    }

    # Make a POST request to the API endpoint
    response = client.post("/review_resume", json=request_payload)

    # Assert the status code is 200 OK
    assert response.status_code == 200

    # Assert the response body matches the expected structure and content
    response_data = response.json()
    assert response_data["review"] == "The resume shows good potential but needs tailoring."
    assert "Strong technical skills" in response_data["strengths"]
    assert "Lack of specific achievements" in response_data["weaknesses"]
    assert "Quantify achievements" in response_data["suggestions"]
    assert "Python" in response_data["matched_keywords"]
    assert "Kubernetes" in response_data["missing_keywords"]

    # Validate the response against the Pydantic model
    try:
        ResumeReviewResponse(**response_data)
    except Exception as e:
        pytest.fail(f"Response does not match Pydantic model: {e}")

@patch('google.generativeai.GenerativeModel')
def test_review_resume_gemini_api_failure(mock_generative_model):
    """
    Test the /review_resume endpoint when the Gemini API call fails.
    Mocks the Gemini API call to raise an exception.
    """
    # Configure the mock object to raise an exception
    mock_instance = mock_generative_model.return_value
    mock_instance.generate_content_async.side_effect = Exception("Gemini API error")

    # Define the request payload
    request_payload = {
        "job_description": "Software Engineer",
        "resume_text": "My resume."
    }

    # Make a POST request to the API endpoint
    response = client.post("/review_resume", json=request_payload)

    # Assert the status code is 500 Internal Server Error
    assert response.status_code == 500
    assert "Failed to review resume" in response.json()["detail"]

@patch('google.generativeai.GenerativeModel')
def test_review_resume_empty_response(mock_generative_model):
    """
    Test the /review_resume endpoint when Gemini returns an empty or malformed response.
    Mocks the Gemini API call to return an empty response.
    """
    # Configure the mock object to return an empty response
    mock_instance = mock_generative_model.return_value
    mock_instance.generate_content_async.return_value = {"candidates": []} # Or any other malformed structure

    # Define the request payload
    request_payload = {
        "job_description": "Data Scientist",
        "resume_text": "My data science resume."
    }

    # Make a POST request to the API endpoint
    response = client.post("/review_resume", json=request_payload)

    # Assert the status code is 500 Internal Server Error
    assert response.status_code == 500
    assert "Gemini API did not return expected content." in response.json()["detail"]

def test_review_resume_invalid_input():
    """
    Test the /review_resume endpoint with invalid input (missing fields).
    """
    # Missing resume_text
    request_payload = {
        "job_description": "Invalid test"
    }

    response = client.post("/review_resume", json=request_payload)
    assert response.status_code == 422 # Unprocessable Entity for Pydantic validation errors
    assert "resume_text" in response.json()["detail"][0]["loc"]

    # Missing job_description
    request_payload = {
        "resume_text": "Invalid test"
    }

    response = client.post("/review_resume", json=request_payload)
    assert response.status_code == 422
    assert "job_description" in response.json()["detail"][0]["loc"]
