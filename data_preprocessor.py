"""
Data preprocessing script for Telugu Health Q&A dataset
Handles data cleaning, tokenization, validation, and format conversion for MT5 training
"""

import json
import re
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
import logging
from pathlib import Path
import argparse
from dataclasses import dataclass
import unicodedata
from collections import Counter
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing"""
    min_question_length: int = 5
    max_question_length: int = 500
    min_answer_length: int = 10
    max_answer_length: int = 1000
    remove_duplicates: bool = True
    validate_telugu: bool = True
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42

class TeluguHealthDataPreprocessor:
    """Main class for preprocessing Telugu health Q&A data"""
    
    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()
        self.telugu_range = '\u0C00-\u0C7F'  # Telugu Unicode range
        self.english_range = 'a-zA-Z'
        
        # Health-related keywords for validation
        self.health_keywords_telugu = [
            'ఆరోగ్యం', 'వైద్యం', 'నొప్పి', 'జ్వరం', 'దగ్గు', 'తలనొప్పి', 'కడుపునొప్పి',
            'వెన్నునొప్పి', 'మధుమేహం', 'రక్తపోటు', 'నిద్రలేకపోవడం', 'వైద్యుడు',
            'మందు', 'చికిత్స', 'వ్యాధి', 'లక్షణం', 'వైద్య', 'ఆస్పత్రి', 'డాక్టర్',
            'నర్స్', 'మాత్రలు', 'ఇంజెక్షన్', 'ఆపరేషన్', 'పరీక్ష', 'రిపోర్ట్'
        ]
        
        self.health_keywords_english = [
            'health', 'medicine', 'pain', 'fever', 'cough', 'headache', 'stomach',
            'diabetes', 'pressure', 'doctor', 'treatment', 'disease', 'symptom',
            'hospital', 'medical', 'pharmacy', 'medication', 'prescription'
        ]
        
        # Common Telugu health terms mapping
        self.health_term_mapping = {
            'తలనొప్పి': 'headache',
            'కడుపునొప్పి': 'stomach ache',
            'వెన్నునొప్పి': 'back pain',
            'జ్వరం': 'fever',
            'దగ్గు': 'cough',
            'మధుమేహం': 'diabetes',
            'రక్తపోటు': 'blood pressure',
            'నిద్రలేకపోవడం': 'insomnia',
            'వైద్యుడు': 'doctor',
            'మందు': 'medicine',
            'చికిత్స': 'treatment'
        }
        
        random.seed(self.config.random_seed)
    
    def load_data(self, file_path: str) -> List[Dict]:
        """Load data from various file formats"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() == '.json':
            return self._load_json(file_path)
        elif file_path.suffix.lower() == '.csv':
            return self._load_csv(file_path)
        elif file_path.suffix.lower() in ['.txt', '.tsv']:
            return self._load_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _load_json(self, file_path: Path) -> List[Dict]:
        """Load data from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                if 'data' in data:
                    return data['data']
                elif 'questions' in data:
                    return data['questions']
                else:
                    # Convert dict to list format
                    return [{'question': k, 'answer': v} for k, v in data.items()]
            else:
                raise ValueError("Invalid JSON structure")
                
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {e}")
    
    def _load_csv(self, file_path: Path) -> List[Dict]:
        """Load data from CSV file"""
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            
            # Try different column name variations
            question_cols = ['question', 'Question', 'q', 'Q', 'query', 'Query']
            answer_cols = ['answer', 'Answer', 'a', 'A', 'response', 'Response']
            
            question_col = None
            answer_col = None
            
            for col in question_cols:
                if col in df.columns:
                    question_col = col
                    break
            
            for col in answer_cols:
                if col in df.columns:
                    answer_col = col
                    break
            
            if not question_col or not answer_col:
                raise ValueError(f"Could not find question/answer columns. Available columns: {list(df.columns)}")
            
            return df[[question_col, answer_col]].rename(
                columns={question_col: 'question', answer_col: 'answer'}
            ).to_dict('records')
            
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")
    
    def _load_txt(self, file_path: Path) -> List[Dict]:
        """Load data from text file (tab-separated or custom format)"""
        data = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Try different separators
                if '\t' in line:
                    parts = line.split('\t', 1)
                elif '|' in line:
                    parts = line.split('|', 1)
                elif '::' in line:
                    parts = line.split('::', 1)
                else:
                    logger.warning(f"Could not parse line {i+1}: {line[:50]}...")
                    continue
                
                if len(parts) == 2:
                    data.append({
                        'question': parts[0].strip(),
                        'answer': parts[1].strip()
                    })
            
            return data
            
        except Exception as e:
            raise ValueError(f"Error loading text file: {e}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text or not isinstance(text, str):
            return ""
        
        # Unicode normalization
        text = unicodedata.normalize('NFC', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Telugu, English, and basic punctuation
        text = re.sub(r'[^\u0C00-\u0C7Fa-zA-Z0-9\s\.\?\!\,\;\:\-\(\)\[\]\"\']+', '', text)
        
        # Fix common punctuation issues
        text = re.sub(r'\.{2,}', '.', text)  # Multiple dots
        text = re.sub(r'\?{2,}', '?', text)  # Multiple question marks
        text = re.sub(r'!{2,}', '!', text)   # Multiple exclamation marks
        
        # Ensure proper sentence ending
        text = text.strip()
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text
    
    def is_telugu_text(self, text: str, min_telugu_ratio: float = 0.3) -> bool:
        """Check if text contains sufficient Telugu characters"""
        if not text:
            return False
        
        # Count Telugu characters
        telugu_chars = len(re.findall(f'[{self.telugu_range}]', text))
        total_chars = len(re.sub(r'\s', '', text))  # Exclude spaces
        
        if total_chars == 0:
            return False
        
        telugu_ratio = telugu_chars / total_chars
        return telugu_ratio >= min_telugu_ratio
    
    def is_health_related(self, text: str) -> bool:
        """Check if text is health-related"""
        text_lower = text.lower()
        
        # Check Telugu health keywords
        telugu_matches = sum(1 for keyword in self.health_keywords_telugu if keyword in text_lower)
        
        # Check English health keywords
        english_matches = sum(1 for keyword in self.health_keywords_english if keyword in text_lower)
        
        # Consider health-related if at least one keyword is found
        return (telugu_matches + english_matches) > 0
    
    def validate_qa_pair(self, item: Dict) -> Tuple[bool, List[str]]:
        """Validate a single Q&A pair"""
        errors = []
        
        # Check required fields
        if 'question' not in item or 'answer' not in item:
            errors.append("Missing 'question' or 'answer' field")
            return False, errors
        
        question = str(item.get('question', '')).strip()
        answer = str(item.get('answer', '')).strip()
        
        # Length validation
        if len(question) < self.config.min_question_length:
            errors.append(f"Question too short (min {self.config.min_question_length} chars)")
        
        if len(question) > self.config.max_question_length:
            errors.append(f"Question too long (max {self.config.max_question_length} chars)")
        
        if len(answer) < self.config.min_answer_length:
            errors.append(f"Answer too short (min {self.config.min_answer_length} chars)")
        
        if len(answer) > self.config.max_answer_length:
            errors.append(f"Answer too long (max {self.config.max_answer_length} chars)")
        
        # Telugu validation
        if self.config.validate_telugu:
            if not self.is_telugu_text(question):
                errors.append("Question does not contain sufficient Telugu text")
            
            if not self.is_telugu_text(answer):
                errors.append("Answer does not contain sufficient Telugu text")
        
        # Health relevance check
        if not self.is_health_related(question) and not self.is_health_related(answer):
            errors.append("Content does not appear to be health-related")
        
        return len(errors) == 0, errors
    
    def remove_duplicates(self, data: List[Dict]) -> List[Dict]:
        """Remove duplicate Q&A pairs"""
        seen = set()
        unique_data = []
        
        for item in data:
            question = item.get('question', '').strip().lower()
            answer = item.get('answer', '').strip().lower()
            
            # Create a hash of the Q&A pair
            qa_hash = hash((question, answer))
            
            if qa_hash not in seen:
                seen.add(qa_hash)
                unique_data.append(item)
        
        removed_count = len(data) - len(unique_data)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate Q&A pairs")
        
        return unique_data
    
    def augment_data(self, data: List[Dict], augmentation_factor: float = 0.2) -> List[Dict]:
        """Simple data augmentation for Telugu health Q&A"""
        augmented_data = data.copy()
        
        # Number of samples to augment
        num_to_augment = int(len(data) * augmentation_factor)
        
        for _ in range(num_to_augment):
            # Select random sample
            original = random.choice(data)
            
            # Simple augmentation strategies
            augmented_question = self._augment_question(original['question'])
            augmented_answer = self._augment_answer(original['answer'])
            
            if augmented_question and augmented_answer:
                augmented_data.append({
                    'question': augmented_question,
                    'answer': augmented_answer
                })
        
        logger.info(f"Generated {num_to_augment} augmented samples")
        return augmented_data
    
    def _augment_question(self, question: str) -> str:
        """Augment question with simple variations"""
        # Add variation prefixes/suffixes
        variations = [
            f"దయచేసి చెప్పండి, {question}",
            f"{question} - ఏం చేయాలి?",
            f"నాకు తెలియజేయండి: {question}",
            f"{question} గురించి వివరించండి"
        ]
        
        return random.choice(variations)
    
    def _augment_answer(self, answer: str) -> str:
        """Augment answer with simple variations"""
        # Add helpful prefixes
        prefixes = [
            "సాధారణంగా, ",
            "వైద్య సలహా ప్రకారం, ",
            "ఈ సమస్యకు, ",
            "తరచుగా, "
        ]
        
        # Add advice suffixes
        suffixes = [
            " అవసరమైతే వైద్యుడిని సంప్రదించండి.",
            " మరింత వివరాలకు వైద్య నిపుణుడిని సంప్రదించండి.",
            " ఆరోగ్య సమస్యలకు వైద్య సలహా అవసరం."
        ]
        
        prefix = random.choice(prefixes) if random.random() < 0.5 else ""
        suffix = random.choice(suffixes) if random.random() < 0.3 else ""
        
        return f"{prefix}{answer}{suffix}".strip()
    
    def split_data(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split data into train, validation, and test sets"""
        # Shuffle data
        shuffled_data = data.copy()
        random.shuffle(shuffled_data)
        
        total_samples = len(shuffled_data)
        train_size = int(total_samples * self.config.train_split)
        val_size = int(total_samples * self.config.val_split)
        
        train_data = shuffled_data[:train_size]
        val_data = shuffled_data[train_size:train_size + val_size]
        test_data = shuffled_data[train_size + val_size:]
        
        logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def generate_statistics(self, data: List[Dict]) -> Dict:
        """Generate comprehensive statistics about the dataset"""
        if not data:
            return {}
        
        questions = [item.get('question', '') for item in data]
        answers = [item.get('answer', '') for item in data]
        
        # Length statistics
        q_lengths = [len(q) for q in questions]
        a_lengths = [len(a) for a in answers]
        
        # Word count statistics
        q_word_counts = [len(q.split()) for q in questions]
        a_word_counts = [len(a.split()) for a in answers]
        
        # Health term frequency
        health_term_freq = Counter()
        for text in questions + answers:
            for term in self.health_keywords_telugu:
                health_term_freq[term] += text.lower().count(term)
        
        statistics = {
            'total_samples': len(data),
            'question_stats': {
                'avg_length': sum(q_lengths) / len(q_lengths),
                'min_length': min(q_lengths),
                'max_length': max(q_lengths),
                'avg_words': sum(q_word_counts) / len(q_word_counts)
            },
            'answer_stats': {
                'avg_length': sum(a_lengths) / len(a_lengths),
                'min_length': min(a_lengths),
                'max_length': max(a_lengths),
                'avg_words': sum(a_word_counts) / len(a_word_counts)
            },
            'health_terms': dict(health_term_freq.most_common(10)),
            'telugu_ratio': sum(self.is_telugu_text(q) for q in questions) / len(questions)
        }
        
        return statistics
    
    def preprocess_dataset(self, input_file: str, output_dir: str = "./processed_data") -> Dict:
        """Main preprocessing pipeline"""
        logger.info(f"Starting preprocessing of {input_file}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load data
        data = self.load_data(input_file)
        logger.info(f"Loaded {len(data)} samples")
        
        # Clean and validate data
        valid_data = []
        validation_errors = []
        
        for i, item in enumerate(data):
            # Clean text
            if 'question' in item:
                item['question'] = self.clean_text(item['question'])
            if 'answer' in item:
                item['answer'] = self.clean_text(item['answer'])
            
            # Validate
            is_valid, errors = self.validate_qa_pair(item)
            if is_valid:
                valid_data.append(item)
            else:
                validation_errors.append({'index': i, 'errors': errors, 'item': item})
        
        logger.info(f"Valid samples after cleaning: {len(valid_data)}")
        logger.info(f"Invalid samples: {len(validation_errors)}")
        
        # Remove duplicates
        if self.config.remove_duplicates:
            valid_data = self.remove_duplicates(valid_data)
        
        # Data augmentation (optional)
        # valid_data = self.augment_data(valid_data, 0.1)
        
        # Split data
        train_data, val_data, test_data = self.split_data(valid_data)
        
        # Save processed data
        datasets = {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }
        
        for split_name, split_data in datasets.items():
            output_file = output_path / f"{split_name}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(split_data)} samples to {output_file}")
        
        # Save validation errors
        if validation_errors:
            error_file = output_path / "validation_errors.json"
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(validation_errors, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(validation_errors)} validation errors to {error_file}")
        
        # Generate and save statistics
        stats = self.generate_statistics(valid_data)
        stats_file = output_path / "dataset_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info("Preprocessing completed successfully!")
        
        return {
            'total_processed': len(valid_data),
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data),
            'validation_errors': len(validation_errors),
            'statistics': stats,
            'output_dir': str(output_path)
        }

def create_sample_dataset_file(output_file: str = "sample_telugu_health_qa.json"):
    """Create a sample Telugu health Q&A dataset file"""
    sample_data = [
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
        },
        {
            "question": "డయాబెటిస్ ఎలా నియంత్రించాలి?",
            "answer": "డయాబెటిస్ నియంత్రణకు సమతుల్య ఆహారం, నిత్య వ్యాయామం, మందుల సక్రమ సేవన అవసరం. చక్కెర స్థాయిలను రోజువారీ పరిశీలించండి మరియు వైద్య సలహాలను పాటించండి."
        },
        {
            "question": "అధిక బరువు తగ్గడానికి ఏమి చేయాలి?",
            "answer": "బరువు తగ్గడానికి కెలోరీలను తగ్గించండి, ఫైబర్ అధికంగా ఉన్న ఆహారాలు తీసుకోండి, రోజువారీ వ్యాయామం చేయండి మరియు తగినంత నీరు త్రాగండి. వేగవంతమైన బరువు తగ్గింపు కోసం వైద్యుడిని సంప్రదించండి."
        },
        {
            "question": "హైపర్ టెన్షన్ లక్షణాలు ఏమిటి?",
            "answer": "హైపర్ టెన్షన్ లక్షణాలు: తలనొప్పి, మెడ నొప్పి, చెవుల్లో గణగణ శబ్దం, మూర్ఛ వస్తుంది. తరచుగా ఎలాంటి లక్షణాలు కనిపించవు కాబట్టి రోజూ BP చెక్ చేయించుకోవాలి."
        },
        {
            "question": "గర్భిణీ స్త్రీలు ఏమి జాగ్రత్తలు తీసుకోవాలి?",
            "answer": "గర్భిణీ స్త్రీలు పోషకాహారం తీసుకోవాలి, ఫోలిక్ యాసిడ్ టాబ్లెట్లు వాడాలి, రెగ్యులర్ చెక్అప్లు చేయించుకోవాలి. దైనందిన వ్యాయామం చేయాలి మరియు మద్యం, ధూమపానం మానుకోవాలి."
        }
    ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Created sample dataset with {len(sample_data)} samples: {output_file}")
    return sample_data

def main():
    """Command-line interface for data preprocessing"""
    parser = argparse.ArgumentParser(description="Preprocess Telugu Health Q&A dataset for MT5 training")
    parser.add_argument("input_file", help="Input dataset file (JSON, CSV, or TXT)")
    parser.add_argument("--output_dir", default="./processed_data", help="Output directory for processed data")
    parser.add_argument("--min_question_length", type=int, default=5, help="Minimum question length")
    parser.add_argument("--max_question_length", type=int, default=500, help="Maximum question length")
    parser.add_argument("--min_answer_length", type=int, default=10, help="Minimum answer length")
    parser.add_argument("--max_answer_length", type=int, default=1000, help="Maximum answer length")
    parser.add_argument("--train_split", type=float, default=0.8, help="Training set ratio")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--no_duplicates", action="store_true", help="Remove duplicate Q&A pairs")
    parser.add_argument("--no_telugu_validation", action="store_true", help="Skip Telugu text validation")
    parser.add_argument("--create_sample", action="store_true", help="Create sample dataset file")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_dataset_file()
        return
    
    # Create preprocessing configuration
    config = PreprocessingConfig(
        min_question_length=args.min_question_length,
        max_question_length=args.max_question_length,
        min_answer_length=args.min_answer_length,
        max_answer_length=args.max_answer_length,
        train_split=args.train_split,
        val_split=args.val_split,
        remove_duplicates=not args.no_duplicates,
        validate_telugu=not args.no_telugu_validation
    )
    
    # Initialize preprocessor
    preprocessor = TeluguHealthDataPreprocessor(config)
    
    # Process dataset
    try:
        results = preprocessor.preprocess_dataset(args.input_file, args.output_dir)
        
        print("\n" + "="*50)
        print("PREPROCESSING COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Total processed samples: {results['total_processed']}")
        print(f"Training samples: {results['train_samples']}")
        print(f"Validation samples: {results['val_samples']}")
        print(f"Test samples: {results['test_samples']}")
        print(f"Validation errors: {results['validation_errors']}")
        print(f"Output directory: {results['output_dir']}")
        print("\nDataset Statistics:")
        stats = results['statistics']
        print(f"  Average question length: {stats['question_stats']['avg_length']:.1f} chars")
        print(f"  Average answer length: {stats['answer_stats']['avg_length']:.1f} chars")
        print(f"  Telugu text ratio: {stats['telugu_ratio']:.2%}")
        print(f"  Top health terms: {list(stats['health_terms'].keys())[:5]}")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())