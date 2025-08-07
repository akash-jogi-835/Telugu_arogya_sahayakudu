import random

class TeluguHealthQA:
    """
    Mock Telugu Health Q&A system that simulates MT5 fine-tuned model responses
    """
    
    def __init__(self):
        # Predefined Telugu health Q&A pairs for simulation
        self.qa_pairs = [
            {
                "question": "తలనొప్పికి ఏమి చేయాలి?",
                "answer": "తలనొప్పికి విశ్రాంతి తీసుకోండి, నీరు ఎక్కువగా త్రాగండి మరియు అవసరమైతే పారాసిటమాల్ తీసుకోండి. నొప్పి కొనసాగితే వైద్యుడిని సంప్రదించండి."
            },
            {
                "question": "జ్వరం వచ్చినప్పుడు ఏమి చేయాలి?",
                "answer": "జ్వరం వచ్చినప్పుడు విశ్రాంతి తీసుకోండి, ద్రవాలు ఎక్కువగా త్రాగండి మరియు చల్లని వస్త్రంతో శరీరాన్ని తుడుచుకోండి. 102°F కంటే ఎక్కువ ఉంటే వైద్యుడిని సంప్రదించండి."
            },
            {
                "question": "కడుపునొప్పికి ఇంటి వైద్యం ఏమిటి?",
                "answer": "కడుపునొప్పికి అల్లం టీ త్రాగండి, తేలికపాటి ఆహారం తీసుకోండి మరియు వేడిమిని కడుపుపై పెట్టండి. నొప్పి తీవ్రంగా ఉంటే వైద్య సహాయం తీసుకోండి."
            },
            {
                "question": "దగ్గుకు ఏమి చేయాలి?",
                "answer": "దగ్గుకు తేనె మరియు అల్లం కలిపిన వేడిమైన నీరు త్రాగండి, ఆవిరి పీల్చుకోండి మరియు తగినంత విశ్రాంతి తీసుకోండి. రెండు వారాలకంటే ఎక్కువ కొనసాగితే వైద్యుడిని సంప్రదించండి."
            },
            {
                "question": "మధుమేహం ఉన్నవారు ఏమి తినాలి?",
                "answer": "మధుమేహం ఉన్నవారు తక్కువ గ్లైసెమిక్ ఇండెక్స్ ఉన్న ఆహారాలు తీసుకోవాలి. కూరగాయలు, ధాన్యాలు, మరియు ప్రోటీన్ ఆహారాలు మంచివి. చక్కెర మరియు మిఠాయిలను తగ్గించండి."
            },
            {
                "question": "రక్తపోటు ఎక్కువగా ఉంటే ఏమి చేయాలి?",
                "answer": "రక్తపోటు ఎక్కువగా ఉంటే ఉప్పు తక్కువగా తీసుకోండి, నిత్యం వ్యాయామం చేయండి, మరియు ఒత్తిడిని తగ్గించండి. వైద్యుడి సూచన ప్రకారం మందులు తీసుకోండి."
            },
            {
                "question": "నిద్రలేకపోవడానికి ఏమి చేయాలి?",
                "answer": "నిద్రలేకపోవడానికి రోజువారీ వ్యాయామం చేయండి, కెఫీన్ తగ్గించండి మరియు నిద్రకు ముందు రిలాక్సేషన్ టెక్నిక్స్ చేయండి. నిద్రకు అనుకూలమైన వాతావరణం సృష్టించండి."
            },
            {
                "question": "వెన్నునొప్పికి ఏమి చేయాలి?",
                "answer": "వెన్నునొప్పికి వేడిమిని లేదా చల్లదనాన్ని ప్రయోగించండి, తేలికపాటి వ్యాయామాలు చేయండి మరియు సరైన భంగిమలో కూర్చోండి. తీవ్రమైన నొప్పికి భౌతిక చికిత్స తీసుకోండి."
            }
        ]
        
        # Common health topics for pattern matching
        self.health_topics = {
            "తలనొప్పి": ["విశ్రాంతి", "నీరు", "పారాసిటమాల్", "వైద్యుడిని సంప్రదించండి"],
            "జ్వరం": ["విశ్రాంతి", "ద్రవాలు", "చల్లని వస్త్రం", "వైద్య సహాయం"],
            "కడుపునొప్పి": ["అల్లం టీ", "తేలికపాటి ఆహారం", "వేడిమి", "వైద్య సహాయం"],
            "దగ్గు": ["తేనె", "అల్లం", "ఆవిరి", "విశ్రాంతి"],
            "మధుమేహం": ["తక్కువ గ్లైసెమిక్", "కూరగాయలు", "ప్రోటీన్", "చక్కెర తగ్గించండి"],
            "రక్తపోటు": ["ఉప్పు తగ్గించండి", "వ్యాయామం", "ఒత్తిడి తగ్గించండి", "మందులు"],
            "నిద్రలేకపోవడం": ["వ్యాయామం", "కెఫీన్ తగ్గించండి", "రిలాక్సేషన్", "నిద్ర వాతావరణం"],
            "వెన్నునొప్పి": ["వేడిమి లేదా చల్లదనం", "వ్యాయామాలు", "సరైన భంగిమ", "భౌతిక చికిత్స"]
        }
    
    def get_answer(self, question):
        """
        Generate an answer for the given Telugu health question
        """
        # First, try to find exact match
        for qa in self.qa_pairs:
            if self._similarity_check(question, qa["question"]) > 0.7:
                return qa["answer"]
        
        # If no exact match, generate pattern-based response
        return self._generate_pattern_response(question)
    
    def _similarity_check(self, q1, q2):
        """
        Simple similarity check based on common words
        """
        words1 = set(q1.lower().split())
        words2 = set(q2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _generate_pattern_response(self, question):
        """
        Generate response based on identified health topics
        """
        question_lower = question.lower()
        
        # Check for known health topics
        for topic, keywords in self.health_topics.items():
            if any(word in question_lower for word in topic.lower().split()):
                response_parts = random.sample(keywords, min(3, len(keywords)))
                return f"{topic}కి సంబంధించి: " + ", ".join(response_parts) + ". వైద్య సలహా కోసం డాక్టరును సంప్రదించండి."
        
        # Generic health response
        generic_responses = [
            "మీ ఆరోగ్య సమస్యకు సంబంధించి వైద్యుడిని సంప్రదించడం మంచిది. సాధారణంగా విశ్రాంతి మరియు తగిన ఆహారం అవసరం.",
            "ఈ లక్షణాలకు అనేక కారణాలు ఉండవచ్చు. వైద్య పరీక్ష చేయించుకోవడం మంచిది. అదనంగా ఆరోగ్యకరమైన జీవనశైలిని అనుసరించండి.",
            "మీ ప్రశ్నకు సరైన సమాధానం కోసం వైద్య నిపుణుడిని సంప్రదించండి. అప్పటి వరకు విశ్రాంతి తీసుకుని ఆరోగ్యకరమైన ఆహారం తీసుకోండి."
        ]
        
        return random.choice(generic_responses)
    
    def get_sample_questions(self):
        """
        Return sample questions for demonstration
        """
        return [qa["question"] for qa in self.qa_pairs[:5]]
    
    def get_health_categories(self):
        """
        Return available health categories
        """
        return list(self.health_topics.keys())
