import os
import google.generativeai as genai

# Configure API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# List available models
response = genai.models.list()  # use .models.list(), not client.list_models()
for model in response.models:
    print(model.name, "-", model.type)
