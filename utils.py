import json
from difflib import SequenceMatcher

def load_knowledge_base(path="cancer_knowledge_base.json"):
    """Load the knowledge base from a JSON file."""
    try:
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Knowledge base file '{path}' not found.")
    except json.JSONDecodeError:
        print(f" Error decoding JSON from '{path}'.")

    return {}

def similarity(a, b):
    """Calculate similarity between two strings using SequenceMatcher."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def extract_keywords(text):
    """Extract cancer-related keywords from user input."""
    cancer_keywords = [
        'cancer', 'tumor', 'malignant', 'benign', 'metastasis', 'oncology',
        'chemotherapy', 'radiation', 'surgery', 'treatment', 'symptoms',
        'diagnosis', 'prevention', 'screening', 'risk', 'stage', 'grade'
    ]

    cancer_types = [
        'breast', 'lung', 'prostate', 'colorectal', 'skin', 'melanoma',
        'brain', 'liver', 'kidney', 'bladder', 'pancreatic', 'ovarian',
        'cervical', 'thyroid', 'leukemia', 'lymphoma', 'sarcoma'
    ]

    text_lower = text.lower()
    return [kw for kw in cancer_keywords + cancer_types if kw in text_lower]

def match_intent(user_input, cancer_data, similarity_threshold=0.6):
    """
    Match user input to the knowledge base (intents or cancer-type format).
    Returns the best matching response or a fallback.
    """
    if not cancer_data or not user_input:
        return get_default_response()

    user_input_clean = user_input.lower().strip()

    # Strategy 1: Intents-based matching
    if 'intents' in cancer_data:
        for intent in cancer_data['intents']:
            tag = intent.get('tag', '')
            patterns = intent.get('patterns', [])
            answer = intent.get('answer', '')

            # Direct pattern match using string similarity
            for pattern in patterns:
                if similarity(user_input_clean, pattern.lower()) >= similarity_threshold:
                    return answer

            # Keyword-based loose match
            user_keywords = extract_keywords(user_input)
            if any(
                kw in tag.lower() or any(kw in pattern.lower() for pattern in patterns)
                for kw in user_keywords
            ):
                return answer

    # Strategy 2: Cancer-type structure
    for cancer_type, info in cancer_data.items():
        if cancer_type == 'intents':
            continue

        if isinstance(info, dict) and cancer_type.lower() in user_input_clean:
            for intent, responses in info.items():
                if any(keyword in user_input_clean for keyword in intent.lower().split()):
                    if isinstance(responses, list) and responses:
                        return responses[0]
                    elif isinstance(responses, str):
                        return responses

            # Generic fallback for that cancer type
            if 'general' in info:
                general = info['general']
                return general[0] if isinstance(general, list) else general
            else:
                return f"I can help you with questions about {cancer_type}. Please be more specific."

    # Strategy 3: Extracted keyword acknowledgment
    user_keywords = extract_keywords(user_input)
    if user_keywords:
        return f"I see you're asking about {', '.join(user_keywords)}. Could you please clarify your question?"

    # Strategy 4: Vague question detection
    question_types = {
        'what': ['definition', 'general'],
        'how': ['treatment', 'prevention', 'diagnosis'],
        'why': ['causes', 'risk factors'],
        'when': ['symptoms', 'screening'],
        'where': ['spread', 'location']
    }

    for qtype in question_types:
        if user_input_clean.startswith(qtype):
            return f"It seems you're asking a '{qtype}' type question. Please include the specific cancer or issue."

    return get_default_response()

def get_cancer_types():
    """Return a list of supported cancer types."""
    return [
        'breast cancer', 'lung cancer', 'prostate cancer', 'colorectal cancer',
        'skin cancer', 'melanoma', 'brain tumor', 'liver cancer', 'kidney cancer',
        'bladder cancer', 'pancreatic cancer', 'ovarian cancer', 'cervical cancer',
        'thyroid cancer', 'leukemia', 'lymphoma', 'sarcoma'
    ]

def format_response(response, max_length=500):
    """Trim long responses and return a clean reply."""
    if not response:
        return "I’m sorry, I don’t have information on that topic yet."

    response = response.strip()
    return response[:max_length].rsplit(" ", 1)[0] + "..." if len(response) > max_length else response

def get_default_response():
    """Standard fallback response for unmatched questions."""
    return (
        "I’m trained to assist with cancer-related topics. Please ask me about symptoms, "
        "treatment options, prevention methods, or specific cancer types. For personal medical concerns, "
        "please consult a licensed healthcare provider."
    )
