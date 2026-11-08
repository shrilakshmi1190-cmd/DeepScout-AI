import os
from dotenv import load_dotenv

load_dotenv()
from llm.gemini_client import generate_answer as generate_gemini
from llm.openai_client import generate_answer as generate_openai
from llm.local_client import generate_answer as generate_local
from llm.mock_llm import generate_answer as generate_mock


def generate_answer(prompt):

    provider = os.getenv("LLM_PROVIDER", "gemini")

    if provider == "gemini":
        return generate_gemini(prompt)

    elif provider == "openai":
        return generate_openai(prompt)

    elif provider == "local":
        return generate_local(prompt)

    elif provider == "mock":
        return generate_mock(prompt)

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}"
        )