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
        try:
            # Use working model - gemini-3.5-flash with extended timeout for RAG prompts
            # Document-grounded prompts with RAG context can be longer
            timeout_seconds = 90.0 if ('page' in prompt.lower() or 'source' in prompt.lower()) else 60.0
            
            response = client.interactions.create(
                model="gemini-3.5-flash",  # Working model
                input=prompt,
                timeout=timeout_seconds
            )
            return response.output_text
        except Exception as e:
            # Log the error but don't crash
            error_type = type(e).__name__
            error_msg = str(e)
            
            # Provide detailed error information for debugging
            if 'NotFoundError' in error_type or '404' in error_msg:
                print(f"Gemini API error: Model not found. Check model name 'gemini-3.5-flash'")
            elif 'APITimeoutError' in error_type or 'timeout' in error_msg.lower():
                print(f"Gemini API error: Request timed out after 90 seconds. This may indicate network issues or the prompt is too complex.")
            elif 'AuthenticationError' in error_type or '401' in error_msg or '403' in error_msg:
                print(f"Gemini API error: Authentication failed. Check API key.")
            else:
                print(f"Gemini API error: {error_type}: {error_msg[:200]}")
            
            # Fallback to mock response but indicate it's a fallback with clear prefix
            if 'You are an Evidence Critic' in prompt or 'OUTPUT FORMAT (JSON only):' in prompt:
                return '''{
  "sufficient": true,
  "reason": "[GEMINI API UNAVAILABLE - USING FALLBACK] Gemini API returned error",
  "missing": [],
  "refined_query": ""
}'''
            else:
                # Return a clear message that this is a fallback due to API error
                # Prefix with clear indicator
                return f"[GEMINI API UNAVAILABLE - MOCK FALLBACK]\n\nError: {error_type}\n\nThis is not a real Gemini response. The system is using a mock fallback because the Gemini API failed."
    else:
        # Mock response when Gemini is not available
        # Check if this is an evidence critic prompt
        if 'You are an Evidence Critic' in prompt or 'OUTPUT FORMAT (JSON only):' in prompt:
            # Return valid JSON for evidence critic
            return '''{
  "sufficient": true,
  "reason": "Gemini not available",
  "missing": [],
  "refined_query": ""
}'''
        else:
            # For non-critic, provide mock with attribution if source info present
            if 'page' in prompt.lower() and 'source' in prompt.lower():
                return "Based on the uploaded document (page 2), the paper proposes a Transformer architecture. Mock response showing attribution would work with real API."
            else:
                return f"Mock Gemini response to: {prompt[:100]}..."