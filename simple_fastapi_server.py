"""
Simple FastAPI server for Telugu Health Q&A
Uses the simplified trainer for serving predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import torch
import json
import logging
from pathlib import Path
import time
from datetime import datetime

from simple_mt5_trainer import SimpleTrainer, SimpleSeq2SeqModel, SimpleTokenizer, create_sample_training_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Telugu Health Q&A API",
    description="Simple REST API for Telugu health question answering",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
trainer: Optional[SimpleTrainer] = None
model_loaded = False

class HealthQuestion(BaseModel):
    question: str
    max_length: Optional[int] = 100

class HealthAnswer(BaseModel):
    question: str
    answer: str
    confidence: float
    processing_time: float
    timestamp: str

class TrainingRequest(BaseModel):
    dataset: List[Dict[str, str]]

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global trainer, model_loaded
    
    logger.info("Starting Telugu Health Q&A API...")
    
    # Try to load existing model
    model_path = "./models/simple_telugu_health_qa.pt"
    if Path(model_path).exists():
        try:
            await load_existing_model(model_path)
        except Exception as e:
            logger.warning(f"Failed to load existing model: {e}")
            await initialize_new_model()
    else:
        await initialize_new_model()

async def load_existing_model(model_path: str):
    """Load existing trained model"""
    global trainer, model_loaded
    
    # Create tokenizer and model
    tokenizer = SimpleTokenizer()
    
    # Load checkpoint to get vocab info
    checkpoint = torch.load(model_path, map_location='cpu')
    tokenizer.char_to_idx = checkpoint['tokenizer_char_to_idx']
    tokenizer.idx_to_char = checkpoint['tokenizer_idx_to_char']
    tokenizer.vocab_size = checkpoint['vocab_size']
    
    model = SimpleSeq2SeqModel(tokenizer.vocab_size)
    trainer = SimpleTrainer(model, tokenizer)
    trainer.load_model(model_path)
    
    model_loaded = True
    logger.info("Existing model loaded successfully")

async def initialize_new_model():
    """Initialize and train a new model"""
    global trainer, model_loaded
    
    logger.info("Initializing new model with sample data...")
    
    # Create training data
    train_data = create_sample_training_data()
    
    # Build tokenizer
    tokenizer = SimpleTokenizer()
    all_texts = []
    for item in train_data:
        all_texts.append(item['question'])
        all_texts.append(item['answer'])
    tokenizer.build_vocab(all_texts)
    
    # Create model and trainer
    model = SimpleSeq2SeqModel(tokenizer.vocab_size)
    trainer = SimpleTrainer(model, tokenizer)
    
    # Quick training
    trainer.train(train_data, epochs=5, batch_size=2, lr=0.001)
    
    # Save model
    Path("./models").mkdir(exist_ok=True)
    trainer.save_model("./models/simple_telugu_health_qa.pt")
    
    model_loaded = True
    logger.info("New model trained and saved successfully")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_loaded
    }

@app.post("/ask", response_model=HealthAnswer)
async def ask_health_question(question_request: HealthQuestion):
    """Generate answer for Telugu health question"""
    global trainer, model_loaded
    
    if not model_loaded or trainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Generate answer
        answer = trainer.generate_answer(question_request.question)
        
        # Calculate confidence (simple heuristic)
        confidence = min(0.95, max(0.60, len(answer) / 50))
        
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

@app.post("/train")
async def train_model(training_request: TrainingRequest):
    """Train model with provided dataset"""
    global trainer, model_loaded
    
    if len(training_request.dataset) < 3:
        raise HTTPException(status_code=400, detail="Dataset must contain at least 3 Q&A pairs")
    
    try:
        # Validate dataset format
        for i, item in enumerate(training_request.dataset):
            if "question" not in item or "answer" not in item:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Item {i} missing 'question' or 'answer' field"
                )
        
        # Build tokenizer
        tokenizer = SimpleTokenizer()
        all_texts = []
        for item in training_request.dataset:
            all_texts.append(item['question'])
            all_texts.append(item['answer'])
        tokenizer.build_vocab(all_texts)
        
        # Create model and trainer
        model = SimpleSeq2SeqModel(tokenizer.vocab_size)
        trainer = SimpleTrainer(model, tokenizer)
        
        # Train
        trainer.train(training_request.dataset, epochs=5, batch_size=2, lr=0.001)
        
        # Save model
        trainer.save_model("./models/simple_telugu_health_qa.pt")
        
        model_loaded = True
        
        return {
            "message": "Training completed successfully",
            "total_samples": len(training_request.dataset),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/sample-questions")
async def get_sample_questions():
    """Get sample Telugu health questions"""
    sample_questions = [
        "తలనొప్పికి ఏమి చేయాలి?",
        "జ్వరం వచ్చినప్పుడు ఏమి చేయాలి?",
        "కడుపునొప్పికి ఏమి చేయాలి?",
        "దగ్గుకు ఏమి చేయాలి?",
        "మధుమేహం ఉన్నవారు ఏమి తినాలి?",
        "రక్తపోటు ఎక్కువగా ఉంటే ఏమి చేయాలి?",
        "నిద్రలేకపోవడానికి ఏమి చేయాలి?",
        "వెన్నునొప్పికి ఏమి చేయాలి?"
    ]
    
    return {"sample_questions": sample_questions}

@app.get("/model-status")
async def get_model_status():
    """Get current model status"""
    return {
        "model_loaded": model_loaded,
        "model_path": "./models/simple_telugu_health_qa.pt" if model_loaded else None,
        "last_updated": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "simple_fastapi_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )