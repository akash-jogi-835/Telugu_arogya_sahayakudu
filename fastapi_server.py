"""
FastAPI server for Telugu Health Q&A model serving
Provides REST API endpoints for health question answering using fine-tuned MT5 model
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
import logging
import json
import os
from pathlib import Path
import time
from datetime import datetime
import asyncio
import aiofiles

from mt5_fine_tuner import MT5FineTuner, TrainingConfig, create_sample_dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Telugu Health Q&A API",
    description="REST API for Telugu health question answering using fine-tuned MT5 model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
fine_tuner: Optional[MT5FineTuner] = None
model_loaded = False
training_status = {"is_training": False, "progress": 0, "message": ""}

# Pydantic models for API requests/responses
class HealthQuestion(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="Telugu health question")
    max_length: Optional[int] = Field(100, ge=10, le=300, description="Maximum length of generated answer")
    
class HealthAnswer(BaseModel):
    question: str
    answer: str
    confidence: float
    processing_time: float
    timestamp: str

class TrainingRequest(BaseModel):
    dataset: List[Dict[str, str]] = Field(..., description="List of Q&A pairs for training")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Training configuration")

class TrainingResponse(BaseModel):
    message: str
    training_id: str
    status: str

class ModelStatus(BaseModel):
    model_loaded: bool
    model_path: Optional[str]
    training_status: Dict[str, Any]
    last_updated: str

class EvaluationMetrics(BaseModel):
    bleu_score: float
    rouge_1: float
    rouge_2: float
    rouge_l: float
    accuracy: float
    total_samples: int

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global fine_tuner, model_loaded
    
    logger.info("Starting Telugu Health Q&A API...")
    
    # Try to load existing model
    model_path = "./mt5_telugu_health_output/best_model.pt"
    if os.path.exists(model_path):
        try:
            config = TrainingConfig()
            fine_tuner = MT5FineTuner(config)
            fine_tuner.load_model(model_path)
            model_loaded = True
            logger.info("Pre-trained model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load pre-trained model: {e}")
            model_loaded = False
    else:
        logger.info("No pre-trained model found. Model will be trained on first request.")
        model_loaded = False

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_loaded
    }

# Main Q&A endpoint
@app.post("/ask", response_model=HealthAnswer)
async def ask_health_question(question_request: HealthQuestion):
    """
    Generate answer for Telugu health question
    """
    global fine_tuner, model_loaded
    
    start_time = time.time()
    
    # Check if model is loaded  
    if not model_loaded or fine_tuner is None:
        # Try to initialize with sample data if no model exists
        try:
            await initialize_model_with_sample_data()
        except Exception as e:
            raise HTTPException(
                status_code=503, 
                detail=f"Model not available. Please train the model first. Error: {str(e)}"
            )
    
    try:
        # Generate answer
        answer = fine_tuner.generate_answer(
            question_request.question, 
            max_length=question_request.max_length
        )
        
        # Calculate confidence (mock for now)
        confidence = min(0.95, max(0.60, len(answer) / 100))  # Simple heuristic
        
        processing_time = time.time() - start_time
        
        return HealthAnswer(
            question=question_request.question,
            answer=answer,
            confidence=confidence,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(e)}")

# Batch Q&A endpoint
@app.post("/ask-batch")
async def ask_batch_questions(questions: List[HealthQuestion]):
    """
    Process multiple health questions in batch
    """
    if len(questions) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 10 questions")
    
    results = []
    for question_request in questions:
        try:
            result = await ask_health_question(question_request)
            results.append(result)
        except Exception as e:
            results.append({
                "question": question_request.question,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    return {"results": results, "total": len(results)}

# Training endpoint
@app.post("/train", response_model=TrainingResponse)
async def train_model(training_request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Start model training with provided dataset
    """
    global training_status
    
    if training_status["is_training"]:
        raise HTTPException(status_code=409, detail="Training is already in progress")
    
    if len(training_request.dataset) < 5:
        raise HTTPException(status_code=400, detail="Dataset must contain at least 5 Q&A pairs")
    
    # Validate dataset format
    for i, item in enumerate(training_request.dataset):
        if "question" not in item or "answer" not in item:
            raise HTTPException(
                status_code=400, 
                detail=f"Item {i} missing 'question' or 'answer' field"
            )
    
    # Generate training ID
    training_id = f"training_{int(time.time())}"
    
    # Start training in background
    background_tasks.add_task(run_training, training_request.dataset, training_request.config, training_id)
    
    return TrainingResponse(
        message="Training started successfully",
        training_id=training_id,
        status="started"
    )

async def run_training(dataset: List[Dict], config_dict: Optional[Dict], training_id: str):
    """
    Run model training in background
    """
    global fine_tuner, model_loaded, training_status
    
    try:
        training_status = {"is_training": True, "progress": 0, "message": "Initializing training..."}
        
        # Create training configuration
        if config_dict:
            config = TrainingConfig(**config_dict)
        else:
            config = TrainingConfig(
                batch_size=4,
                num_epochs=3,
                learning_rate=1e-4,
                output_dir=f"./models/{training_id}"
            )
        
        # Initialize trainer
        fine_tuner = MT5FineTuner(config)
        
        training_status = {"is_training": True, "progress": 10, "message": "Preparing data..."}
        
        # Prepare data
        fine_tuner.prepare_data(dataset)
        
        training_status = {"is_training": True, "progress": 20, "message": "Starting training..."}
        
        # Train model
        fine_tuner.train()
        
        training_status = {"is_training": False, "progress": 100, "message": "Training completed successfully"}
        model_loaded = True
        
        logger.info(f"Training {training_id} completed successfully")
        
    except Exception as e:
        training_status = {"is_training": False, "progress": 0, "message": f"Training failed: {str(e)}"}
        logger.error(f"Training {training_id} failed: {e}")

# Training status endpoint
@app.get("/training-status")
async def get_training_status():
    """
    Get current training status
    """
    return training_status

# Model status endpoint
@app.get("/model-status", response_model=ModelStatus)
async def get_model_status():
    """
    Get current model status
    """
    model_path = None
    if model_loaded and fine_tuner:
        model_path = fine_tuner.config.output_dir
    
    return ModelStatus(
        model_loaded=model_loaded,
        model_path=model_path,
        training_status=training_status,
        last_updated=datetime.now().isoformat()
    )

# Dataset upload endpoint
@app.post("/upload-dataset")
async def upload_dataset(dataset: List[Dict[str, str]]):
    """
    Upload and validate dataset for training
    """
    if len(dataset) < 5:
        raise HTTPException(status_code=400, detail="Dataset must contain at least 5 Q&A pairs")
    
    # Validate dataset
    errors = []
    for i, item in enumerate(dataset):
        if "question" not in item:
            errors.append(f"Item {i}: Missing 'question' field")
        if "answer" not in item:
            errors.append(f"Item {i}: Missing 'answer' field")
        if "question" in item and len(item["question"].strip()) < 5:
            errors.append(f"Item {i}: Question too short")
        if "answer" in item and len(item["answer"].strip()) < 10:
            errors.append(f"Item {i}: Answer too short")
    
    if errors:
        raise HTTPException(status_code=400, detail={"message": "Dataset validation failed", "errors": errors})
    
    # Save dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dataset_{timestamp}.json"
    
    async with aiofiles.open(f"./datasets/{filename}", "w", encoding="utf-8") as f:
        await f.write(json.dumps(dataset, ensure_ascii=False, indent=2))
    
    return {
        "message": "Dataset uploaded and validated successfully",
        "filename": filename,
        "total_samples": len(dataset),
        "timestamp": datetime.now().isoformat()
    }

# Evaluation endpoint
@app.post("/evaluate", response_model=EvaluationMetrics)
async def evaluate_model(test_dataset: List[Dict[str, str]]):
    """
    Evaluate model performance on test dataset
    """
    global fine_tuner, model_loaded
    
    if not model_loaded or fine_tuner is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(test_dataset) < 3:
        raise HTTPException(status_code=400, detail="Test dataset must contain at least 3 samples")
    
    try:
        # Simple evaluation using string similarity and length matching
        from rouge_score import rouge_scorer
        import sacrebleu
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        total_samples = len(test_dataset)
        rouge_1_scores = []
        rouge_2_scores = []
        rouge_l_scores = []
        bleu_scores = []
        exact_matches = 0
        
        for item in test_dataset:
            question = item["question"]
            reference_answer = item["answer"]
            
            # Generate prediction
            predicted_answer = fine_tuner.generate_answer(question, max_length=150)
            
            # Calculate ROUGE scores
            rouge_scores = scorer.score(reference_answer, predicted_answer)
            rouge_1_scores.append(rouge_scores['rouge1'].fmeasure)
            rouge_2_scores.append(rouge_scores['rouge2'].fmeasure)
            rouge_l_scores.append(rouge_scores['rougeL'].fmeasure)
            
            # Calculate BLEU score
            bleu_score = sacrebleu.sentence_bleu(predicted_answer, [reference_answer]).score / 100.0
            bleu_scores.append(bleu_score)
            
            # Check exact match (simple)
            if predicted_answer.strip().lower() == reference_answer.strip().lower():
                exact_matches += 1
        
        return EvaluationMetrics(
            bleu_score=sum(bleu_scores) / len(bleu_scores),
            rouge_1=sum(rouge_1_scores) / len(rouge_1_scores),
            rouge_2=sum(rouge_2_scores) / len(rouge_2_scores),
            rouge_l=sum(rouge_l_scores) / len(rouge_l_scores),
            accuracy=exact_matches / total_samples,
            total_samples=total_samples
        )
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

# Sample questions endpoint
@app.get("/sample-questions")
async def get_sample_questions():
    """
    Get sample Telugu health questions for testing
    """
    sample_questions = [
        "తలనొప్పికి ఏమి చేయాలి?",
        "జ్వరం వచ్చినప్పుడు ఏమి చేయాలి?",
        "కడుపునొప్పికి ఇంటి వైద్యం ఏమిటి?",
        "దగ్గుకు ఏమి చేయాలి?",
        "మధుమేహం ఉన్నవారు ఏమి తినాలి?",
        "రక్తపోటు ఎక్కువగా ఉంటే ఏమి చేయాలి?",
        "నిద్రలేకపోవడానికి ఏమి చేయాలి?",
        "వెన్నునొప్పికి ఏమి చేయాలి?",
        "డయాబెటిస్ ఎలా నియంత్రించాలి?",
        "అధిక బరువు తగ్గడానికి ఏమి చేయాలి?"
    ]
    
    return {"sample_questions": sample_questions}

async def initialize_model_with_sample_data():
    """
    Initialize model with sample data if no trained model exists
    """
    global fine_tuner, model_loaded
    
    logger.info("Initializing model with sample data...")
    
    config = TrainingConfig(
        batch_size=2,
        num_epochs=1,
        learning_rate=1e-4,
        output_dir="./mt5_telugu_health_sample"
    )
    
    fine_tuner = MT5FineTuner(config)
    sample_data = create_sample_dataset()
    
    fine_tuner.prepare_data(sample_data)
    fine_tuner.train()
    
    model_loaded = True
    logger.info("Model initialized with sample data")

# Create necessary directories
os.makedirs("./datasets", exist_ok=True)
os.makedirs("./models", exist_ok=True)

if __name__ == "__main__":
    uvicorn.run(
        "fastapi_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )