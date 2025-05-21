# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import os
import uvicorn
import json # Import json module
import re   # Import regex module

# Initialize FastAPI app
app = FastAPI(
    title="Resume Review API",
    description="An API to review resumes against job descriptions using Gemini Flash 2.5."
)

# Pydantic model for the request body
class ResumeReviewRequest(BaseModel):
    job_description: str
    resume_text: str

# Pydantic model for the response body
class ResumeReviewResponse(BaseModel):
    review: str
    strengths: list[str]
    weaknesses: list[str]
    suggestions: list[str]
    matched_keywords: list[str]
    missing_keywords: list[str]

@app.on_event("startup")
async def startup_event():
    """
    Initialize the Gemini API key on application startup.
    The API key is expected to be available in the environment.
    For Canvas, the API key is automatically provided in the fetch call.
    """
    api_key = os.getenv("GEMINI_API_KEY", "") # In Canvas, this will be empty, and the runtime provides the key.
    if not api_key:
        print("Warning: GEMINI_API_KEY environment variable not set. Using default for local testing.")
    genai.configure(api_key=api_key)

async def parse_gemini_response(response_text: str) -> dict:
    """Parse the Gemini API response text into a dictionary."""
    match = re.search(r'```json\s*(\{.*\})\s*```', response_text, re.DOTALL)
    json_string = match.group(1) if match else response_text.strip()
    return json.loads(json_string)

async def generate_gemini_review(job_description: str, resume_text: str) -> dict:
    """Generate a resume review using Gemini AI."""
    prompt = f"""
    You are an expert resume reviewer. Your task is to analyze a resume against a given job description.
    Provide a comprehensive review, highlighting strengths, weaknesses, and actionable suggestions.
    Also, identify keywords from the job description that are present in the resume and those that are missing.

    Job Description:
    ---
    {job_description}
    ---

    Resume Text:
    ---
    {resume_text}
    ---

    Please provide the review strictly in a structured JSON format.
    DO NOT include any introductory or concluding text, or any markdown code block delimiters (e.g., ```json).
    The output should be ONLY the JSON object.

    Use the following keys for the JSON object:
    - "review": A general summary of the resume's alignment with the job description.
    - "strengths": A list of key strengths of the resume in relation to the job description.
    - "weaknesses": A list of key weaknesses or areas for improvement.
    - "suggestions": A list of actionable suggestions to improve the resume.
    - "matched_keywords": A list of important keywords from the job description found in the resume.
    - "missing_keywords": A list of important keywords from the job description that are missing from the resume.
    """

    model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
    response = await model.generate_content_async(prompt)

    has_candidates = bool(response.candidates)
    has_valid_content = has_candidates and response.candidates[0].content and response.candidates[0].content.parts
    if not has_valid_content:
        raise HTTPException(status_code=500, detail="Gemini API did not return expected content.")

    raw_response_text = response.candidates[0].content.parts[0].text
    print(f"Raw Gemini response: {raw_response_text}")
    return await parse_gemini_response(raw_response_text)

@app.post("/review_resume", response_model=ResumeReviewResponse)
async def review_resume(request: ResumeReviewRequest):
    """
    Reviews a given resume against a job description using the Gemini Flash 2.5 model.

    Args:
        request (ResumeReviewRequest): A request object containing the job description and resume text.

    Returns:
        ResumeReviewResponse: A structured review of the resume.

    Raises:
        HTTPException: If there's an error calling the Gemini API.
    """
    try:
        review_data = await generate_gemini_review(request.job_description, request.resume_text)
        return ResumeReviewResponse(**review_data)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from Gemini API response: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to parse Gemini API response as JSON: {str(e)}")
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to review resume: {str(e)}")

# This block is for running the app directly with `python main.py`
# For Docker, uvicorn will be called directly.
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
