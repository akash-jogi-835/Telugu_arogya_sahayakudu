import re

class TeluguTextProcessor:
    """
    Basic text processing utilities for Telugu text
    """
    
    def __init__(self):
        # Telugu Unicode ranges
        self.telugu_range = '\u0C00-\u0C7F'
        
        # Common Telugu health-related terms
        self.health_terms = {
            'నొప్పి': 'pain',
            'జ్వరం': 'fever',
            'దగ్గు': 'cough',
            'తలనొప్పి': 'headache',
            'కడుపునొప్పి': 'stomach_ache',
            'వెన్నునొప్పి': 'back_pain',
            'మధుమేహం': 'diabetes',
            'రక్తపోటు': 'blood_pressure',
            'నిద్రలేకపోవడం': 'insomnia',
            'వైద్యుడు': 'doctor',
            'మందు': 'medicine',
            'చికిత్స': 'treatment',
            'ఆరోగ్యం': 'health',
            'వ్యాధి': 'disease',
            'లక్షణం': 'symptom'
        }
    
    def preprocess(self, text):
        """
        Basic preprocessing for Telugu text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep Telugu characters and basic punctuation
        text = re.sub(r'[^\u0C00-\u0C7F\s\?\!\.\,\-]', '', text)
        
        # Normalize question marks
        text = re.sub(r'[？]', '?', text)
        
        return text
    
    def is_telugu_text(self, text):
        """
        Check if text contains Telugu characters
        """
        if not text:
            return False
        
        telugu_pattern = f'[{self.telugu_range}]'
        telugu_chars = re.findall(telugu_pattern, text)
        
        # Consider text as Telugu if at least 30% characters are Telugu
        return len(telugu_chars) / len(text.replace(' ', '')) > 0.3
    
    def extract_health_terms(self, text):
        """
        Extract health-related terms from Telugu text
        """
        if not text:
            return []
        
        found_terms = []
        text_lower = text.lower()
        
        for telugu_term, english_term in self.health_terms.items():
            if telugu_term in text_lower:
                found_terms.append({
                    'telugu': telugu_term,
                    'english': english_term,
                    'position': text_lower.find(telugu_term)
                })
        
        # Sort by position in text
        found_terms.sort(key=lambda x: x['position'])
        
        return found_terms
    
    def clean_response(self, response):
        """
        Clean and format Telugu response text
        """
        if not response:
            return ""
        
        # Remove extra whitespace
        response = re.sub(r'\s+', ' ', response.strip())
        
        # Ensure proper sentence ending
        if not response.endswith(('.', '!', '?')):
            response += '.'
        
        # Capitalize first letter if it's English
        if response and response[0].isalpha():
            response = response[0].upper() + response[1:]
        
        return response
    
    def get_question_type(self, question):
        """
        Identify the type of health question
        """
        if not question:
            return "unknown"
        
        question_lower = question.lower()
        
        # Question type patterns
        question_types = {
            "symptom": ["లక్షణం", "సంకేతం", "ఎలా తెలుసుకోవాలి"],
            "treatment": ["చికిత్స", "ఏమి చేయాలి", "వైద్యం", "మందు"],
            "prevention": ["నివారణ", "ఎలా నివారించాలి", "జాగ్రత్తలు"],
            "cause": ["కారణం", "ఎందుకు వస్తుంది", "ఎందుకు అవుతుంది"],
            "medication": ["మందు", "మాత్రలు", "డోసేజ్", "ఎంత తీసుకోవాలి"],
            "diet": ["ఆహారం", "తిండి", "ఏమి తినాలి", "డైట్"],
            "general": ["ఏమిటి", "ఎలా", "ఎప్పుడు"]
        }
        
        for q_type, patterns in question_types.items():
            if any(pattern in question_lower for pattern in patterns):
                return q_type
        
        return "general"
    
    def validate_input(self, text):
        """
        Validate Telugu health question input
        """
        errors = []
        
        if not text or not text.strip():
            errors.append("Please enter a question")
            return False, errors
        
        if len(text.strip()) < 5:
            errors.append("Question is too short")
        
        if len(text.strip()) > 500:
            errors.append("Question is too long (max 500 characters)")
        
        if not self.is_telugu_text(text):
            errors.append("Please enter your question in Telugu")
        
        # Check for inappropriate content (basic check)
        inappropriate_patterns = ['పేజీ', 'లింక్', 'http', 'www']
        if any(pattern in text.lower() for pattern in inappropriate_patterns):
            errors.append("Please enter a valid health question")
        
        return len(errors) == 0, errors
    
    def get_word_count(self, text):
        """
        Count words in Telugu text
        """
        if not text:
            return 0
        
        # Split by whitespace and filter empty strings
        words = [word for word in text.split() if word.strip()]
        return len(words)
    
    def get_character_stats(self, text):
        """
        Get character statistics for Telugu text
        """
        if not text:
            return {
                'total_chars': 0,
                'telugu_chars': 0,
                'spaces': 0,
                'punctuation': 0
            }
        
        telugu_pattern = f'[{self.telugu_range}]'
        telugu_chars = len(re.findall(telugu_pattern, text))
        spaces = text.count(' ')
        punctuation = len(re.findall(r'[^\w\s]', text))
        
        return {
            'total_chars': len(text),
            'telugu_chars': telugu_chars,
            'spaces': spaces,
            'punctuation': punctuation,
            'telugu_percentage': (telugu_chars / len(text.replace(' ', ''))) * 100 if text.replace(' ', '') else 0
        }
