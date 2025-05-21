# test_main.py
from fastapi.testclient import TestClient
from main import app, ResumeReviewResponse
import pytest
from unittest.mock import patch, AsyncMock, MagicMock # Import MagicMock
import json

# Create a TestClient instance for the FastAPI app
client = TestClient(app)

@pytest.fixture
def mock_gemini_response_object():
    """
    Fixture to provide a mocked Gemini API response object,
    mimicking the structure returned by the actual genai.GenerativeModel.
    """
    json_content = {
        "review": "The resume shows good potential but needs tailoring.",
        "strengths": ["Strong technical skills", "Relevant project experience"],
        "weaknesses": ["Lack of specific achievements", "Generic summary"],
        "suggestions": ["Quantify achievements", "Tailor summary to job description"],
        "matched_keywords": ["Python", "FastAPI", "Docker"],
        "missing_keywords": ["Kubernetes", "CI/CD"]
    }
    # Simulate the markdown wrapped JSON output that the model might return
    mock_text_response = f"```json\n{json.dumps(json_content, indent=2)}\n```"

    # Create a MagicMock object that mimics the structure of the actual response
    mock_response_object = MagicMock()

    # Set up nested attributes using MagicMock
    # response.candidates[0].content.parts[0].text
    mock_response_object.candidates = [MagicMock()] # List of candidates
    mock_response_object.candidates[0].content = MagicMock() # First candidate has content
    mock_response_object.candidates[0].content.parts = [MagicMock()] # Content has parts
    mock_response_object.candidates[0].content.parts[0].text = mock_text_response # First part has text

    return mock_response_object

@patch('google.generativeai.GenerativeModel')
def test_review_resume_success(mock_generative_model, mock_gemini_response_object): # Use the new fixture name
    """
    Test the /review_resume endpoint for a successful response.
    Mocks the Gemini API call to return a predefined response object.
    """
    mock_instance = mock_generative_model.return_value
    # Use AsyncMock to make the return value awaitable, pointing to the MagicMock object
    mock_instance.generate_content_async = AsyncMock(return_value=mock_gemini_response_object)

    request_payload = {
        "job_description": "We are looking for a Python developer with FastAPI and Docker experience.",
        "resume_text": "Experienced developer with Python and Docker skills. Built APIs."
    }

    response = client.post("/review_resume", json=request_payload)

    assert response.status_code == 200
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
    mock_instance = mock_generative_model.return_value
    mock_instance.generate_content_async = AsyncMock(side_effect=Exception("Gemini API error"))

    request_payload = {
        "job_description": "Software Engineer",
        "resume_text": "My resume."
    }

    response = client.post("/review_resume", json=request_payload)

    assert response.status_code == 500
    assert "Failed to review resume" in response.json()["detail"]

@patch('google.generativeai.GenerativeModel')
def test_review_resume_empty_response(mock_generative_model):
    """
    Test the /review_resume endpoint when Gemini returns an empty or malformed response.
    Mocks the Gemini API call to return an empty response.
    """
    mock_instance = mock_generative_model.return_value
    # The empty response should also be an object with the expected structure, even if empty.
    mock_response_object = MagicMock()
    mock_response_object.candidates = [] # Empty list of candidates
    mock_instance.generate_content_async = AsyncMock(return_value=mock_response_object)

    request_payload = {
        "job_description": "Data Scientist",
        "resume_text": "My data science resume."
    }

    response = client.post("/review_resume", json=request_payload)

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
