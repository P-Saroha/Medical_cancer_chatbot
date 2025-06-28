# chatbot.py
import torch
import json
import pickle
import logging
import numpy as np
import re
import warnings
from typing import Optional, Dict
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from gemini_config import generate_medical_response
from model import Classifier as CancerClassifier, Config

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalChatBot:
    def __init__(self, 
                 kb_path: str = "cancer_knowledge_base.json",
                 model_path: str = "best_medical_model.pth",
                 tag_mappings_path: str = "idx2tag.pickle"):
        self.kb_path = kb_path
        self.model_path = model_path
        self.tag_mappings_path = tag_mappings_path
        self.min_confidence = 0.6
        self.fallback_mode = False

        self.load_knowledge_base()
        self.load_intent_model()
        self.load_semantic_model()

    def load_knowledge_base(self):
        try:
            with open(self.kb_path, 'r', encoding='utf-8') as f:
                kb_json = json.load(f)
            self.kb_entries = kb_json['intents']
            self.pattern_to_entry = {}
            self.all_patterns = []

            for entry in self.kb_entries:
                for pattern in entry.get("patterns", []):
                    self.all_patterns.append(pattern)
                    self.pattern_to_entry[pattern] = entry

            logger.info(f"Knowledge base loaded with {len(self.kb_entries)} intents")
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
            self.kb_entries = []

    def load_intent_model(self):
        try:
            with open(self.tag_mappings_path, 'rb') as f:
                self.idx_to_tag = pickle.load(f)

            config = Config()
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.intent_model = CancerClassifier(len(self.idx_to_tag)).to(self.device)
            self.intent_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.intent_model.eval()

            logger.info("Intent model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load intent model: {e}")
            self.fallback_mode = True
            self.intent_model = None
            self.tokenizer = None

    def load_semantic_model(self):
        try:
            self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.pattern_embeddings = self.semantic_model.encode(self.all_patterns, show_progress_bar=True)
            logger.info("Semantic model and embeddings ready")
        except Exception as e:
            logger.error(f"Semantic model error: {e}")
            self.semantic_model = None

    def is_medical_question(self, text: str) -> bool:
        medical_keywords = [
            'cancer', 'tumor', 'symptom', 'chemo', 'radiation', 'diagnosis', 'oncology',
            'biopsy', 'treatment', 'metastasis', 'oncologist', 'side effects', 'malignant'
        ]
        return any(word in text.lower() for word in medical_keywords)

    def get_kb_response(self, query: str) -> Optional[Dict]:
        for pattern, entry in self.pattern_to_entry.items():
            if pattern.lower() in query.lower():
                return {"answer": entry['answer'], "source": "knowledge_base", "confidence": 1.0}

        if hasattr(self, 'semantic_model'):
            query_emb = self.semantic_model.encode([query])
            sims = cosine_similarity(query_emb, self.pattern_embeddings)[0]
            best_idx = np.argmax(sims)
            best_score = sims[best_idx]
            print(f"Semantic similarity score: {best_score:.2f} for '{query}'")
            if best_score > 0.85:
                pattern = self.all_patterns[best_idx]
                entry = self.pattern_to_entry[pattern]
                return {"answer": entry['answer'], "source": "semantic_search", "confidence": best_score}

        return None

    def get_gemini_response(self, query: str) -> str:
        return generate_medical_response(query)

    def get_response(self, query: str, show_debug: bool = False) -> str:
        if not self.is_medical_question(query):
            return ("I'm specialized in cancer and medical topics. Please ask something related to cancer, treatment, symptoms, or diagnosis.")

        kb_result = self.get_kb_response(query)
        if kb_result:
            response = kb_result['answer']
            if show_debug:
                response += f"\n\n[Source: {kb_result['source']} | Confidence: {kb_result['confidence']:.2f}]"
            return response

        gemini_response = self.get_gemini_response(query)
        if show_debug:
            gemini_response += "\n\n[Source: Gemini AI]"
        return gemini_response
