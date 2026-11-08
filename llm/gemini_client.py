import os
from dotenv import load_dotenv

load_dotenv()

# Try to import genai, but provide fallback if not available
try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("Warning: google.ai.genai not available, using mock response")

api_key = os.getenv("GEMINI_API_KEY")

if GENAI_AVAILABLE:
    client = genai.Client(api_key=api_key)
else:
    client = None


def generate_answer(prompt):
    if GENAI_AVAILABLE and client:
        response = client.interactions.create(
            model="gemini-3.7-flash",
            input=prompt
        )
        return response.output_text
    else:
        # Mock response when Gemini is not available
        # Check if this is an evidence critic prompt
        if 'You are an Evidence Critic' in prompt or 'OUTPUT FORMAT (JSON only):' in prompt:
            # Return valid JSON for evidence critic
            return '''{
  "sufficient": true,
  "reason": "",
  "missing": [],
  "refined_query": ""
}'''
        else:
            return f"Mock Gemini response to: {prompt[:100]}..."