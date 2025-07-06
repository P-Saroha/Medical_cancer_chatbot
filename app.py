from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import re

# Import chatbot and Gemini fallback
from chatbot import MedicalChatBot
from gemini_config import generate_medical_response

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
CORS(app)

# Initialize chatbot
try:
    chatbot = MedicalChatBot()
except Exception as e:
    print(f"Failed to initialize chatbot: {e}")
    chatbot = None

# Validate input
def validate_input(text: str) -> tuple[bool, str]:
    if not text or not text.strip():
        return False, "Message cannot be empty"
    if len(text.strip()) < 3:
        return False, "Message too short"
    if len(text) > 1000:
        return False, "Message too long"
    return True, ""

# Clean input
def sanitize_input(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)
    return re.sub(r'\s+', ' ', text).strip()

@app.route('/')
def home():
    return render_template('index.html')  # templates/index.html required

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True)
        print("DEBUG: Received data:", data)

        user_input = data.get("message", "").strip()
        print("DEBUG: User input =", user_input)

        # Validate
        valid, err = validate_input(user_input)
        if not valid:
            return jsonify({"error": err}), 400

        user_input = sanitize_input(user_input)

        # Reject non-medical queries
        if not chatbot or not chatbot.is_medical_question(user_input):
            return jsonify({
                "response": (
                    "I'm specialized in cancer-related medical questions. "
                    "Please ask about a medical topic."
                ),
                "is_medical": False
            })

        # 1. KB match
        kb_response = chatbot.get_kb_response(user_input)
        if kb_response:
            return jsonify({
                "response": kb_response['answer'],
                "source": kb_response['source']
            })

        # 2. BERT intent classification
        predicted_tag = chatbot.predict_intent_with_bert(user_input)
        if predicted_tag:
            for entry in chatbot.kb_entries:
                if entry["tag"] == predicted_tag:
                    return jsonify({
                        "response": entry["answer"],
                        "source": f"bert ({predicted_tag})"
                    })

        # 3. Gemini fallback
        gemini_answer = generate_medical_response(
            prompt=user_input,
            context="You are a cancer-specialized AI medical assistant."
        )
        return jsonify({
            "response": gemini_answer,
            "source": "gemini"
        })

    except Exception as e:
        print("Error in /chat route:", str(e))
        return jsonify({
            "error": "An internal error occurred.",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"Running on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
