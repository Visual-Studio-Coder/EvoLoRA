import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    print("ERROR: No API key found in .env")
    exit(1)

print(f"API key found: {api_key[:8]}...")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Say hello in JSON format with a message field",
    config=types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=200,
    ),
)

print("SUCCESS:", response.text)
