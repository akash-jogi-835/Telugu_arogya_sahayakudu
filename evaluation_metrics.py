"""
Evaluation metrics script for Telugu Health Q&A models
Evaluates fine-tuned models using BLEU, ROUGE, and custom Telugu-specific metrics
"""

import json
import argparse
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import time
from dataclasses import dataclass
import pandas as pd

# Import evaluation libraries
from rouge_score import rouge_scorer
import sacrebleu
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from mt5_fine_tuner import MT5FineTuner, TrainingConfig
from text_processor import TeluguTextProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """Configuration for model evaluation"""
    model_path: str = ""
    test_data_path: str = ""
    output_dir: str = "./evaluation_results"
    batch_size: int = 1
    max_generation_length: int = 150
    rouge_variants: List[str] = None
    calculate_bertscore: bool = False
    calculate_semantic_similarity: bool = True
    custom_metrics: bool = True
    
    def __post_init__(self):
        if self.rouge_variants is None:
            self.rouge_variants = ['rouge1', 'rouge2', 'rougeL']

class TeluguHealthEvaluator:
    """Main class for evaluating Telugu Health Q&A models"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None
        self.text_processor = TeluguTextProcessor()
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            self.config.rouge_variants, use_stemmer=True
        )
        
        # BLEU smoothing function
        self.bleu_smoother = SmoothingFunction()
        
        # Telugu-specific evaluation components
        self.telugu_health_terms = [
            'తలనొప్పి', 'జ్వరం', 'దగ్గు', 'కడుపునొప్పి', 'వెన్నునొప్పి',
            'మధుమేహం', 'రక్తపోటు', 'నిద్రలేకపోవడం', 'వైద్యుడు', 'మందు',
            'చికిత్స', 'ఆరోగ్యం', 'వ్యాధి', 'లక్షణం'
        ]
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def load_model(self, model_path: str):
        """Load the fine-tuned model"""
        try:
            config = TrainingConfig()
            self.model = MT5FineTuner(config)
            self.model.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_test_data(self, test_data_path: str) -> List[Dict]:
        """Load test dataset"""
        try:
            with open(test_data_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            logger.info(f"Loaded {len(test_data)} test samples")
            return test_data
            
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            raise
    
    def generate_predictions(self, test_data: List[Dict]) -> List[Dict]:
        """Generate predictions for test data"""
        predictions = []
        
        logger.info("Generating predictions...")
        start_time = time.time()
        
        for i, item in enumerate(test_data):
            question = item.get('question', '')
            reference_answer = item.get('answer', '')
            
            try:
                # Generate prediction
                predicted_answer = self.model.generate_answer(
                    question, max_length=self.config.max_generation_length
                )
                
                predictions.append({
                    'question': question,
                    'reference_answer': reference_answer,
                    'predicted_answer': predicted_answer,
                    'sample_id': i
                })
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Generated {i + 1}/{len(test_data)} predictions")
                    
            except Exception as e:
                logger.warning(f"Failed to generate prediction for sample {i}: {e}")
                predictions.append({
                    'question': question,
                    'reference_answer': reference_answer,
                    'predicted_answer': "",
                    'sample_id': i,
                    'error': str(e)
                })
        
        total_time = time.time() - start_time
        logger.info(f"Prediction generation completed in {total_time:.2f} seconds")
        
        return predictions
    
    def calculate_bleu_scores(self, predictions: List[Dict]) -> Dict:
        """Calculate BLEU scores"""
        logger.info("Calculating BLEU scores...")
        
        bleu_scores = []
        bleu_1_scores = []
        bleu_2_scores = []
        bleu_3_scores = []
        bleu_4_scores = []
        
        for pred in predictions:
            if 'error' in pred or not pred['predicted_answer']:
                continue
                
            reference = pred['reference_answer']
            prediction = pred['predicted_answer']
            
            # Tokenize
            ref_tokens = word_tokenize(reference.lower())
            pred_tokens = word_tokenize(prediction.lower())
            
            # Calculate BLEU scores
            try:
                # SacreBLEU (corpus-level will be calculated separately)
                bleu_score = sacrebleu.sentence_bleu(
                    prediction, [reference]
                ).score / 100.0
                
                # NLTK BLEU with different n-grams
                bleu_1 = sentence_bleu(
                    [ref_tokens], pred_tokens, 
                    weights=(1, 0, 0, 0),
                    smoothing_function=self.bleu_smoother.method1
                )
                bleu_2 = sentence_bleu(
                    [ref_tokens], pred_tokens, 
                    weights=(0.5, 0.5, 0, 0),
                    smoothing_function=self.bleu_smoother.method1
                )
                bleu_3 = sentence_bleu(
                    [ref_tokens], pred_tokens, 
                    weights=(0.33, 0.33, 0.33, 0),
                    smoothing_function=self.bleu_smoother.method1
                )
                bleu_4 = sentence_bleu(
                    [ref_tokens], pred_tokens, 
                    weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function=self.bleu_smoother.method1
                )
                
                bleu_scores.append(bleu_score)
                bleu_1_scores.append(bleu_1)
                bleu_2_scores.append(bleu_2)
                bleu_3_scores.append(bleu_3)
                bleu_4_scores.append(bleu_4)
                
            except Exception as e:
                logger.warning(f"BLEU calculation failed for sample: {e}")
                continue
        
        return {
            'bleu_sacre': sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0,
            'bleu_1': sum(bleu_1_scores) / len(bleu_1_scores) if bleu_1_scores else 0,
            'bleu_2': sum(bleu_2_scores) / len(bleu_2_scores) if bleu_2_scores else 0,
            'bleu_3': sum(bleu_3_scores) / len(bleu_3_scores) if bleu_3_scores else 0,
            'bleu_4': sum(bleu_4_scores) / len(bleu_4_scores) if bleu_4_scores else 0,
            'valid_samples': len(bleu_scores)
        }
    
    def calculate_rouge_scores(self, predictions: List[Dict]) -> Dict:
        """Calculate ROUGE scores"""
        logger.info("Calculating ROUGE scores...")
        
        rouge_scores = {variant: [] for variant in self.config.rouge_variants}
        
        for pred in predictions:
            if 'error' in pred or not pred['predicted_answer']:
                continue
                
            reference = pred['reference_answer']
            prediction = pred['predicted_answer']
            
            try:
                scores = self.rouge_scorer.score(reference, prediction)
                
                for variant in self.config.rouge_variants:
                    if variant in scores:
                        rouge_scores[variant].append(scores[variant].fmeasure)
                        
            except Exception as e:
                logger.warning(f"ROUGE calculation failed for sample: {e}")
                continue
        
        # Calculate averages
        rouge_averages = {}
        for variant, scores in rouge_scores.items():
            if scores:
                rouge_averages[f'{variant}_precision'] = sum(s for s in scores) / len(scores)
                rouge_averages[f'{variant}_recall'] = sum(s for s in scores) / len(scores)
                rouge_averages[f'{variant}_fmeasure'] = sum(s for s in scores) / len(scores)
            else:
                rouge_averages[f'{variant}_precision'] = 0
                rouge_averages[f'{variant}_recall'] = 0
                rouge_averages[f'{variant}_fmeasure'] = 0
        
        rouge_averages['valid_samples'] = len(rouge_scores[self.config.rouge_variants[0]])
        return rouge_averages
    
    def calculate_exact_match(self, predictions: List[Dict]) -> Dict:
        """Calculate exact match accuracy"""
        exact_matches = 0
        partial_matches = 0
        valid_samples = 0
        
        for pred in predictions:
            if 'error' in pred or not pred['predicted_answer']:
                continue
                
            reference = pred['reference_answer'].strip().lower()
            prediction = pred['predicted_answer'].strip().lower()
            
            valid_samples += 1
            
            if reference == prediction:
                exact_matches += 1
                partial_matches += 1
            elif reference in prediction or prediction in reference:
                partial_matches += 1
        
        return {
            'exact_match_accuracy': exact_matches / valid_samples if valid_samples > 0 else 0,
            'partial_match_accuracy': partial_matches / valid_samples if valid_samples > 0 else 0,
            'exact_matches': exact_matches,
            'partial_matches': partial_matches,
            'valid_samples': valid_samples
        }
    
    def calculate_telugu_specific_metrics(self, predictions: List[Dict]) -> Dict:
        """Calculate Telugu-specific evaluation metrics"""
        logger.info("Calculating Telugu-specific metrics...")
        
        telugu_term_coverage = []
        telugu_fluency_scores = []
        health_relevance_scores = []
        
        for pred in predictions:
            if 'error' in pred or not pred['predicted_answer']:
                continue
                
            reference = pred['reference_answer']
            prediction = pred['predicted_answer']
            
            # Telugu term coverage
            ref_terms = set()
            pred_terms = set()
            
            for term in self.telugu_health_terms:
                if term in reference.lower():
                    ref_terms.add(term)
                if term in prediction.lower():
                    pred_terms.add(term)
            
            if ref_terms:
                coverage = len(ref_terms.intersection(pred_terms)) / len(ref_terms)
                telugu_term_coverage.append(coverage)
            
            # Telugu text fluency (basic check)
            fluency = self.text_processor.is_telugu_text(prediction)
            telugu_fluency_scores.append(1 if fluency else 0)
            
            # Health relevance
            health_relevance = self.text_processor.extract_health_terms(prediction)
            relevance_score = len(health_relevance) / max(1, len(self.telugu_health_terms))
            health_relevance_scores.append(min(1.0, relevance_score))
        
        return {
            'telugu_term_coverage': sum(telugu_term_coverage) / len(telugu_term_coverage) if telugu_term_coverage else 0,
            'telugu_fluency': sum(telugu_fluency_scores) / len(telugu_fluency_scores) if telugu_fluency_scores else 0,
            'health_relevance': sum(health_relevance_scores) / len(health_relevance_scores) if health_relevance_scores else 0,
            'valid_samples': len(telugu_fluency_scores)
        }
    
    def calculate_length_metrics(self, predictions: List[Dict]) -> Dict:
        """Calculate length-based metrics"""
        length_ratios = []
        avg_ref_length = 0
        avg_pred_length = 0
        valid_samples = 0
        
        for pred in predictions:
            if 'error' in pred or not pred['predicted_answer']:
                continue
                
            ref_len = len(pred['reference_answer'])
            pred_len = len(pred['predicted_answer'])
            
            if ref_len > 0:
                length_ratios.append(pred_len / ref_len)
                avg_ref_length += ref_len
                avg_pred_length += pred_len
                valid_samples += 1
        
        return {
            'avg_length_ratio': sum(length_ratios) / len(length_ratios) if length_ratios else 0,
            'avg_reference_length': avg_ref_length / valid_samples if valid_samples > 0 else 0,
            'avg_prediction_length': avg_pred_length / valid_samples if valid_samples > 0 else 0,
            'length_consistency': 1 - abs(1 - (sum(length_ratios) / len(length_ratios))) if length_ratios else 0,
            'valid_samples': valid_samples
        }
    
    def evaluate_model(self) -> Dict:
        """Run complete model evaluation"""
        logger.info("Starting model evaluation...")
        
        # Load model and test data
        if not self.model:
            self.load_model(self.config.model_path)
        
        test_data = self.load_test_data(self.config.test_data_path)
        
        # Generate predictions
        predictions = self.generate_predictions(test_data)
        
        # Save predictions
        predictions_file = Path(self.config.output_dir) / "predictions.json"
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        
        # Calculate all metrics
        evaluation_results = {
            'model_path': self.config.model_path,
            'test_data_path': self.config.test_data_path,
            'total_samples': len(test_data),
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # BLEU scores
        bleu_results = self.calculate_bleu_scores(predictions)
        evaluation_results['bleu'] = bleu_results
        
        # ROUGE scores
        rouge_results = self.calculate_rouge_scores(predictions)
        evaluation_results['rouge'] = rouge_results
        
        # Exact match
        exact_match_results = self.calculate_exact_match(predictions)
        evaluation_results['exact_match'] = exact_match_results
        
        # Length metrics
        length_results = self.calculate_length_metrics(predictions)
        evaluation_results['length_metrics'] = length_results
        
        # Telugu-specific metrics
        if self.config.custom_metrics:
            telugu_results = self.calculate_telugu_specific_metrics(predictions)
            evaluation_results['telugu_metrics'] = telugu_results
        
        # Save evaluation results
        results_file = Path(self.config.output_dir) / "evaluation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Evaluation completed. Results saved to {results_file}")
        
        return evaluation_results
    
    def generate_report(self, evaluation_results: Dict):
        """Generate human-readable evaluation report"""
        report_file = Path(self.config.output_dir) / "evaluation_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("Telugu Health Q&A Model Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Model: {evaluation_results['model_path']}\n")
            f.write(f"Test Data: {evaluation_results['test_data_path']}\n")
            f.write(f"Total Samples: {evaluation_results['total_samples']}\n")
            f.write(f"Evaluation Date: {evaluation_results['evaluation_timestamp']}\n\n")
            
            # BLEU Scores
            f.write("BLEU Scores:\n")
            f.write("-" * 20 + "\n")
            bleu = evaluation_results['bleu']
            f.write(f"SacreBLEU: {bleu['bleu_sacre']:.4f}\n")
            f.write(f"BLEU-1: {bleu['bleu_1']:.4f}\n")
            f.write(f"BLEU-2: {bleu['bleu_2']:.4f}\n")
            f.write(f"BLEU-3: {bleu['bleu_3']:.4f}\n")
            f.write(f"BLEU-4: {bleu['bleu_4']:.4f}\n\n")
            
            # ROUGE Scores
            f.write("ROUGE Scores:\n")
            f.write("-" * 20 + "\n")
            rouge = evaluation_results['rouge']
            for variant in ['rouge1', 'rouge2', 'rougeL']:
                if f'{variant}_fmeasure' in rouge:
                    f.write(f"{variant.upper()}: {rouge[f'{variant}_fmeasure']:.4f}\n")
            f.write("\n")
            
            # Exact Match
            f.write("Accuracy Metrics:\n")
            f.write("-" * 20 + "\n")
            exact = evaluation_results['exact_match']
            f.write(f"Exact Match: {exact['exact_match_accuracy']:.4f}\n")
            f.write(f"Partial Match: {exact['partial_match_accuracy']:.4f}\n\n")
            
            # Telugu Metrics
            if 'telugu_metrics' in evaluation_results:
                f.write("Telugu-Specific Metrics:\n")
                f.write("-" * 30 + "\n")
                telugu = evaluation_results['telugu_metrics']
                f.write(f"Telugu Term Coverage: {telugu['telugu_term_coverage']:.4f}\n")
                f.write(f"Telugu Fluency: {telugu['telugu_fluency']:.4f}\n")
                f.write(f"Health Relevance: {telugu['health_relevance']:.4f}\n\n")
            
            # Length Metrics
            f.write("Length Analysis:\n")
            f.write("-" * 20 + "\n")
            length = evaluation_results['length_metrics']
            f.write(f"Average Length Ratio: {length['avg_length_ratio']:.4f}\n")
            f.write(f"Average Reference Length: {length['avg_reference_length']:.1f} chars\n")
            f.write(f"Average Prediction Length: {length['avg_prediction_length']:.1f} chars\n")
            f.write(f"Length Consistency: {length['length_consistency']:.4f}\n\n")
            
            # Summary
            f.write("Summary:\n")
            f.write("-" * 20 + "\n")
            overall_score = (
                bleu['bleu_sacre'] + 
                rouge.get('rouge1_fmeasure', 0) + 
                rouge.get('rougeL_fmeasure', 0) + 
                exact['exact_match_accuracy']
            ) / 4
            f.write(f"Overall Performance Score: {overall_score:.4f}\n")
            
            if overall_score >= 0.7:
                f.write("Performance: EXCELLENT\n")
            elif overall_score >= 0.5:
                f.write("Performance: GOOD\n")
            elif overall_score >= 0.3:
                f.write("Performance: FAIR\n")
            else:
                f.write("Performance: NEEDS IMPROVEMENT\n")
        
        logger.info(f"Evaluation report saved to {report_file}")

def create_test_dataset():
    """Create a test dataset for evaluation"""
    test_data = [
        {
            "question": "తలనొప్పికి ఏమి చేయాలి?",
            "answer": "తలనొప్పికి విశ్రాంతి తీసుకోండి, నీరు ఎక్కువగా త్రాగండి మరియు అవసరమైతే పారాసిటమాల్ తీసుకోండి."
        },
        {
            "question": "జ్వరం వచ్చినప్పుడు ఏమి చేయాలి?",
            "answer": "జ్వరం వచ్చినప్పుడు విశ్రాంతి తీసుకోండి, ద్రవాలు ఎక్కువగా త్రాగండి మరియు చల్లని వస్త్రంతో శరీరాన్ని తుడుచుకోండి."
        },
        {
            "question": "కడుపునొప్పికి ఇంటి వైద్యం ఏమిటి?",
            "answer": "కడుపునొప్పికి అల్లం టీ త్రాగండి, తేలికపాటి ఆహారం తీసుకోండి మరియు వేడిమిని కడుపుపై పెట్టండి."
        },
        {
            "question": "దగ్గుకు ఏమి చేయాలి?",
            "answer": "దగ్గుకు తేనె మరియు అల్లం కలిపిన వేడిమైన నీరు త్రాగండి, ఆవిరి పీల్చుకోండి మరియు తగినంత విశ్రాంతి తీసుకోండి."
        },
        {
            "question": "మధుమేహం ఉన్నవారు ఏమి తినాలి?",
            "answer": "మధుమేహం ఉన్నవారు తక్కువ గ్లైసెమిక్ ఇండెక్స్ ఉన్న ఆహారాలు తీసుకోవాలి. కూరగాయలు, ధాన్యాలు, మరియు ప్రోటీన్ ఆహారాలు మంచివి."
        }
    ]
    
    with open("test_data.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    logger.info("Test dataset created: test_data.json")

def main():
    """Command-line interface for model evaluation"""
    parser = argparse.ArgumentParser(description="Evaluate Telugu Health Q&A model")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--test_data", required=True, help="Path to test dataset")
    parser.add_argument("--output_dir", default="./evaluation_results", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--max_length", type=int, default=150, help="Maximum generation length")
    parser.add_argument("--create_test", action="store_true", help="Create test dataset")
    
    args = parser.parse_args()
    
    if args.create_test:
        create_test_dataset()
        return
    
    # Create evaluation configuration
    config = EvaluationConfig(
        model_path=args.model_path,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_generation_length=args.max_length
    )
    
    # Initialize evaluator
    evaluator = TeluguHealthEvaluator(config)
    
    try:
        # Run evaluation
        results = evaluator.evaluate_model()
        
        # Generate report
        evaluator.generate_report(results)
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION COMPLETED")
        print("="*60)
        print(f"Model: {results['model_path']}")
        print(f"Test Samples: {results['total_samples']}")
        print(f"BLEU Score: {results['bleu']['bleu_sacre']:.4f}")
        print(f"ROUGE-1: {results['rouge']['rouge1_fmeasure']:.4f}")
        print(f"ROUGE-L: {results['rouge']['rougeL_fmeasure']:.4f}")
        print(f"Exact Match: {results['exact_match']['exact_match_accuracy']:.4f}")
        if 'telugu_metrics' in results:
            print(f"Telugu Fluency: {results['telugu_metrics']['telugu_fluency']:.4f}")
        print(f"Results saved in: {config.output_dir}/")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())