import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

response = client.interactions.create(
    model="gemini-3.7-flash",
    input="""
Explain RAG.

Rules:
1. Explain it in exactly 3 bullet points.
2. Use simple language.
3. Give one real-world analogy.
"""
)

print(response.output_text)