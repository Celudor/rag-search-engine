import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GENAI_MODEL = "gemini-2.5-flash"


def spell_enhancment(query: str) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GENAI_MODEL,
        contents=f"""Fix any spelling errors in this movie search query.
Only correct obvious typos. Don't change correctly spelled words.
Query: "{query}"
If no errors, return the original query.
Corrected:""",
    )
    return response.text
