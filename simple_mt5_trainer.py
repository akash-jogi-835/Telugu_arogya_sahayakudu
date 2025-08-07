"""
Simplified MT5-like trainer for Telugu Health Q&A
Uses basic PyTorch components to create a working seq2seq model
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from typing import List, Dict, Optional
import logging
import random
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleTokenizer:
    """Simple character-level tokenizer for Telugu"""
    
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        self.pad_token_id = 0
        self.start_token_id = 1
        self.end_token_id = 2
        self.unk_token_id = 3
        
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts"""
        chars = set()
        for text in texts:
            chars.update(text)
        
        # Add special tokens
        self.char_to_idx = {
            '<PAD>': self.pad_token_id,
            '<START>': self.start_token_id,
            '<END>': self.end_token_id,
            '<UNK>': self.unk_token_id
        }
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        
        # Add characters
        for i, char in enumerate(sorted(chars), start=4):
            self.char_to_idx[char] = i
            self.idx_to_char[i] = char
        
        self.vocab_size = len(self.char_to_idx)
        logger.info(f"Built vocabulary with {self.vocab_size} tokens")
    
    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Encode text to token IDs"""
        tokens = [self.char_to_idx.get(char, self.unk_token_id) for char in text]
        
        if max_length:
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens.extend([self.pad_token_id] * (max_length - len(tokens)))
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        chars = []
        for token_id in token_ids:
            if token_id == self.pad_token_id:
                break
            if token_id == self.end_token_id:
                break
            char = self.idx_to_char.get(token_id, '<UNK>')
            if char not in ['<PAD>', '<START>', '<END>', '<UNK>']:
                chars.append(char)
        return ''.join(chars)

class TeluguQADataset(Dataset):
    """Dataset for Telugu Q&A pairs"""
    
    def __init__(self, data: List[Dict], tokenizer: SimpleTokenizer, max_length: int = 200):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item.get('question', '')
        answer = item.get('answer', '')
        
        # Encode
        question_tokens = self.tokenizer.encode(question, self.max_length)
        answer_tokens = [self.tokenizer.start_token_id] + self.tokenizer.encode(answer, self.max_length - 1)
        
        # Create target (shifted answer)
        target_tokens = self.tokenizer.encode(answer, self.max_length - 1) + [self.tokenizer.end_token_id]
        if len(target_tokens) < self.max_length:
            target_tokens.extend([self.tokenizer.pad_token_id] * (self.max_length - len(target_tokens)))
        
        return {
            'question': torch.tensor(question_tokens, dtype=torch.long),
            'answer_input': torch.tensor(answer_tokens, dtype=torch.long),
            'answer_target': torch.tensor(target_tokens, dtype=torch.long)
        }

class SimpleSeq2SeqModel(nn.Module):
    """Simple sequence-to-sequence model"""
    
    def __init__(self, vocab_size: int, hidden_size: int = 256, num_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Encoder
        self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        
        # Decoder
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, question, answer_input, hidden=None):
        # Encode question
        question_emb = self.embedding(question)
        encoder_output, hidden = self.encoder(question_emb, hidden)
        
        # Decode answer
        answer_emb = self.embedding(answer_input)
        decoder_output, _ = self.decoder(answer_emb, hidden)
        
        # Apply dropout
        decoder_output = self.dropout(decoder_output)
        
        # Project to vocabulary
        output = self.output_projection(decoder_output)
        
        return output
    
    def generate(self, question, tokenizer, max_length=100):
        """Generate answer for a question"""
        self.eval()
        with torch.no_grad():
            # Encode question
            question_emb = self.embedding(question)
            encoder_output, hidden = self.encoder(question_emb)
            
            # Start decoding
            generated = [tokenizer.start_token_id]
            decoder_input = torch.tensor([[tokenizer.start_token_id]], dtype=torch.long)
            
            for _ in range(max_length):
                decoder_emb = self.embedding(decoder_input)
                decoder_output, hidden = self.decoder(decoder_emb, hidden)
                output = self.output_projection(decoder_output)
                
                # Get next token
                next_token = torch.argmax(output, dim=-1).item()
                
                if next_token == tokenizer.end_token_id:
                    break
                
                generated.append(next_token)
                decoder_input = torch.tensor([[next_token]], dtype=torch.long)
            
            return tokenizer.decode(generated)

class SimpleTrainer:
    """Simple trainer for the seq2seq model"""
    
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
    
    def train(self, train_data: List[Dict], epochs: int = 3, batch_size: int = 4, lr: float = 0.001):
        """Train the model"""
        dataset = TeluguQADataset(train_data, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(dataloader):
                question = batch['question'].to(self.device)
                answer_input = batch['answer_input'].to(self.device)
                answer_target = batch['answer_target'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(question, answer_input)
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), answer_target.reshape(-1))
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 5 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
    
    def generate_answer(self, question: str) -> str:
        """Generate answer for a question"""
        question_tokens = torch.tensor([self.tokenizer.encode(question, 200)], dtype=torch.long).to(self.device)
        answer = self.model.generate(question_tokens, self.tokenizer)
        return answer
    
    def save_model(self, path: str):
        """Save model and tokenizer"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer_char_to_idx': self.tokenizer.char_to_idx,
            'tokenizer_idx_to_char': self.tokenizer.idx_to_char,
            'vocab_size': self.tokenizer.vocab_size
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model and tokenizer"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Restore tokenizer
        self.tokenizer.char_to_idx = checkpoint['tokenizer_char_to_idx']
        self.tokenizer.idx_to_char = checkpoint['tokenizer_idx_to_char']
        self.tokenizer.vocab_size = checkpoint['vocab_size']
        
        # Load model
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")

def create_sample_training_data():
    """Create sample Telugu health Q&A data"""
    return [
        {
            "question": "తలనొప్పికి ఏమి చేయాలి?",
            "answer": "తలనొప్పికి విశ్రాంతి తీసుకోండి మరియు నీరు త్రాగండి."
        },
        {
            "question": "జ్వరం వచ్చినప్పుడు ఏమి చేయాలి?",
            "answer": "జ్వరం వచ్చినప్పుడు విశ్రాంతి తీసుకోండి మరియు ద్రవాలు త్రాగండి."
        },
        {
            "question": "కడుపునొప్పికి ఏమి చేయాలి?",
            "answer": "కడుపునొప్పికి అల్లం టీ త్రాగండి మరియు విశ్రాంతి తీసుకోండి."
        },
        {
            "question": "దగ్గుకు ఏమి చేయాలి?",
            "answer": "దగ్గుకు తేనె మరియు వేడిమైన నీరు త్రాగండి."
        },
        {
            "question": "మధుమేహం ఉన్నవారు ఏమి తినాలి?",
            "answer": "మධుమేహం ఉన్నవారు కూరగాయలు మరియు ధాన్యాలు తినాలి."
        },
        {
            "question": "రక్తపోటు ఎక్కువగా ఉంటే ఏమి చేయాలి?",
            "answer": "రక్తపోటు ఎక్కువగా ఉంటే ఉప్పు తగ్గించండి మరియు వ్యాయామం చేయండి."
        },
        {
            "question": "నిద్రలేకపోవడానికి ఏమి చేయాలి?",
            "answer": "నిద్రలేకపోవడానికి వ్యాయామం చేయండి మరియు కెఫీన్ తగ్గించండి."
        },
        {
            "question": "వెన్నునొప్పికి ఏమి చేయాలి?",
            "answer": "వెన్నునొప్పికి వేడిమి ప్రయోగించండి మరియు వ్యాయామాలు చేయండి."
        }
    ]

def main():
    """Train and test the model"""
    # Create training data
    train_data = create_sample_training_data()
    
    # Build tokenizer
    tokenizer = SimpleTokenizer()
    all_texts = []
    for item in train_data:
        all_texts.append(item['question'])
        all_texts.append(item['answer'])
    tokenizer.build_vocab(all_texts)
    
    # Create model
    model = SimpleSeq2SeqModel(tokenizer.vocab_size)
    
    # Create trainer
    trainer = SimpleTrainer(model, tokenizer)
    
    # Train
    logger.info("Starting training...")
    trainer.train(train_data, epochs=10, batch_size=2, lr=0.001)
    
    # Save model
    Path("./models").mkdir(exist_ok=True)
    trainer.save_model("./models/simple_telugu_health_qa.pt")
    
    # Test generation
    test_questions = [
        "తలనొప్పికి ఏమి చేయాలి?",
        "జ్వరం వచ్చినప్పుడు ఏమి చేయాలి?"
    ]
    
    logger.info("Testing generation...")
    for question in test_questions:
        answer = trainer.generate_answer(question)
        print(f"Q: {question}")
        print(f"A: {answer}")
        print("-" * 50)

if __name__ == "__main__":
    main()