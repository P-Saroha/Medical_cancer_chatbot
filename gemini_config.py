import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GEMINI_MEDICAL_API_KEY = os.getenv("GEMINI_MEDICAL_API_KEY")

class MedicalResponseGenerator:
    def __init__(self):
        genai.configure(api_key=GEMINI_MEDICAL_API_KEY)
        self.model = genai.GenerativeModel(
            model_name="models/gemini-1.5-flash-latest",  
            generation_config=self.get_generation_config(),
            safety_settings=self.get_safety_settings()
        )

    def get_generation_config(self):
        return {
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 20,
            "max_output_tokens": 1024
        }

    def get_safety_settings(self):
        return [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_LOW_AND_ABOVE"}
        ]

    def generate_response(self, prompt: str, context: str = "") -> str:
        try:
            full_prompt = (
                f"Context: {context}\n\n"
                f"Question: {prompt}\n\n"
                "Requirements:\n"
               "Instructions:\n"
                "- Answer in 3–4 sentences only\n"
                "- Be medically accurate and concise\n"
                "- Focus only on cancer-related facts\n"
                "- Avoid unnecessary background or repetition\n"
                "- If unsure, suggest consulting a doctor"

            )
            response = self.model.generate_content(full_prompt)
            return self.validate_response(response.text)
        except Exception as e:
            return f"An error occurred while generating a medical response. Please consult a medical professional\n Gemini API error: {str(e)}"

    def validate_response(self, response: str) -> str:
        if "sorry" in response.lower() or "don't know" in response.lower():
            return "I couldn't find enough medical evidence to answer this. Please consult an oncologist."
        return response

# Singleton usage
medical_response_generator = MedicalResponseGenerator()

def generate_medical_response(prompt: str, context: str = "") -> str:
    return medical_response_generator.generate_response(prompt, context)
