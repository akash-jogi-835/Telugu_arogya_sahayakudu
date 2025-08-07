"""
MT5 Fine-tuning Pipeline for Telugu Health Q&A
This module provides functionality to fine-tune MT5 models for Telugu health question-answering.
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm
from typing import List, Dict, Tuple, Optional
import os
import logging
from pathlib import Path
import pickle
from dataclasses import dataclass
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for MT5 fine-tuning"""
    model_name: str = "mt5-small"
    max_source_length: int = 256
    max_target_length: int = 256
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    save_steps: int = 500
    eval_steps: int = 100
    output_dir: str = "./mt5_telugu_health"
    gradient_accumulation_steps: int = 1
    fp16: bool = False
    dataloader_num_workers: int = 0

class TeluguHealthDataset(Dataset):
    """Dataset class for Telugu Health Q&A pairs"""
    
    def __init__(self, data: List[Dict], tokenizer, config: TrainingConfig):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item.get('question', '')
        answer = item.get('answer', '')
        
        # Add task prefix for MT5
        source_text = f"Telugu health Q&A: {question}"
        target_text = answer
        
        # Tokenize source
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.config.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.config.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': source_encoding['input_ids'].flatten(),
            'attention_mask': source_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten(),
            'target_attention_mask': target_encoding['attention_mask'].flatten()
        }

class SimplifiedMT5Model(nn.Module):
    """Simplified MT5-like model for Telugu health Q&A"""
    
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8, 
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embedding layers
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer layers
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, 
                src_key_padding_mask=None, tgt_key_padding_mask=None):
        """Forward pass"""
        # Embedding
        src_emb = self.encoder_embedding(src) * (self.d_model ** 0.5)
        tgt_emb = self.decoder_embedding(tgt) * (self.d_model ** 0.5)
        
        # Add positional encoding
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        # Generate target mask for autoregressive generation
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))
            if src.is_cuda:
                tgt_mask = tgt_mask.cuda()
        
        # Transformer forward pass
        output = self.transformer(
            src_emb, tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Project to vocabulary
        output = self.output_projection(output)
        
        return output
    
    def generate_square_subsequent_mask(self, sz):
        """Generate mask for autoregressive decoding"""
        mask = torch.triu(torch.ones(sz, sz)) == 1
        mask = mask.transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TeluguTokenizer:
    """Simple tokenizer for Telugu text using SentencePiece"""
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.sp_model = None
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2
        self.bos_token_id = 3
        
    def train_tokenizer(self, texts: List[str], model_prefix: str = "telugu_tokenizer"):
        """Train SentencePiece tokenizer on Telugu texts"""
        # Prepare training data
        with open(f"{model_prefix}_train.txt", "w", encoding="utf-8") as f:
            for text in texts:
                f.write(text + "\n")
        
        # Train SentencePiece model
        spm.SentencePieceTrainer.train(
            input=f"{model_prefix}_train.txt",
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            character_coverage=0.9995,
            model_type='unigram',
            pad_id=self.pad_token_id,
            eos_id=self.eos_token_id,
            unk_id=self.unk_token_id,
            bos_id=self.bos_token_id
        )
        
        # Load the trained model
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(f"{model_prefix}.model")
        
        # Clean up training file
        os.remove(f"{model_prefix}_train.txt")
        
        logger.info(f"Tokenizer trained with vocabulary size: {self.sp_model.vocab_size()}")
    
    def load_tokenizer(self, model_path: str):
        """Load pre-trained tokenizer"""
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(model_path)
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        if self.sp_model is None:
            raise ValueError("Tokenizer not trained or loaded")
        return self.sp_model.encode_as_ids(text)
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        if self.sp_model is None:
            raise ValueError("Tokenizer not trained or loaded")
        return self.sp_model.decode_ids(token_ids)
    
    def __call__(self, text, max_length=None, padding=None, truncation=None, return_tensors=None):
        """Tokenizer call interface similar to transformers"""
        token_ids = self.encode(text)
        
        if truncation and max_length:
            token_ids = token_ids[:max_length]
        
        if padding == 'max_length' and max_length:
            if len(token_ids) < max_length:
                token_ids = token_ids + [self.pad_token_id] * (max_length - len(token_ids))
        
        # Create attention mask
        attention_mask = [1] * len(token_ids)
        if padding == 'max_length' and max_length:
            attention_mask = attention_mask + [0] * (max_length - len(attention_mask))
            attention_mask = attention_mask[:max_length]
        
        result = {
            'input_ids': token_ids,
            'attention_mask': attention_mask
        }
        
        if return_tensors == 'pt':
            result = {k: torch.tensor([v]) for k, v in result.items()}
        
        return result

class MT5FineTuner:
    """Main class for fine-tuning MT5 on Telugu health data"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Using device: {self.device}")
    
    def prepare_data(self, train_data: List[Dict], val_data: Optional[List[Dict]] = None):
        """Prepare training and validation datasets"""
        logger.info("Preparing tokenizer and datasets...")
        
        # Extract all texts for tokenizer training
        all_texts = []
        for item in train_data:
            all_texts.append(item.get('question', ''))
            all_texts.append(item.get('answer', ''))
        
        if val_data:
            for item in val_data:
                all_texts.append(item.get('question', ''))
                all_texts.append(item.get('answer', ''))
        
        # Train tokenizer
        self.tokenizer = TeluguTokenizer(vocab_size=16000)  # Smaller vocab for simplified model
        self.tokenizer.train_tokenizer(all_texts, "telugu_health_tokenizer")
        
        # Create datasets
        self.train_dataset = TeluguHealthDataset(train_data, self.tokenizer, self.config)
        if val_data:
            self.val_dataset = TeluguHealthDataset(val_data, self.tokenizer, self.config)
        else:
            # Split training data for validation
            split_idx = int(0.9 * len(train_data))
            self.val_dataset = TeluguHealthDataset(train_data[split_idx:], self.tokenizer, self.config)
            self.train_dataset = TeluguHealthDataset(train_data[:split_idx], self.tokenizer, self.config)
        
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.val_dataset)}")
    
    def create_model(self):
        """Create and initialize the model"""
        vocab_size = self.tokenizer.sp_model.get_piece_size()
        
        self.model = SimplifiedMT5Model(
            vocab_size=vocab_size,
            d_model=512,
            nhead=8,
            num_encoder_layers=4,  # Smaller for efficiency
            num_decoder_layers=4
        )
        
        self.model.to(self.device)
        logger.info(f"Model created with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def train(self):
        """Train the model"""
        if self.model is None:
            self.create_model()
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.dataloader_num_workers
        )
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.WarmupLR(optimizer, warmup_factor=0.1, warmup_iters=self.config.warmup_steps)
        
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
        # Training loop
        logger.info("Starting training...")
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            total_loss = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                target_attention_mask = batch['target_attention_mask'].to(self.device)
                
                # Create target input (shift labels right)
                tgt_input = labels.clone()
                tgt_input = torch.cat([
                    torch.full((tgt_input.size(0), 1), self.tokenizer.bos_token_id, dtype=tgt_input.dtype, device=self.device),
                    tgt_input[:, :-1]
                ], dim=1)
                
                # Forward pass
                optimizer.zero_grad()
                
                outputs = self.model(
                    src=input_ids,
                    tgt=tgt_input,
                    src_key_padding_mask=(attention_mask == 0),
                    tgt_key_padding_mask=(target_attention_mask == 0)
                )
                
                # Calculate loss
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))
                
                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                global_step += 1
                
                # Logging
                if global_step % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}, Step {global_step}, Loss: {loss.item():.4f}")
                
                # Validation
                if global_step % self.config.eval_steps == 0:
                    val_loss = self.evaluate(val_loader, criterion)
                    logger.info(f"Validation Loss: {val_loss:.4f}")
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save_model(f"{self.config.output_dir}/best_model.pt")
                
                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    self.save_model(f"{self.config.output_dir}/checkpoint_{global_step}.pt")
            
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
        # Save final model
        self.save_model(f"{self.config.output_dir}/final_model.pt")
        logger.info("Training completed!")
    
    def evaluate(self, val_loader, criterion):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                target_attention_mask = batch['target_attention_mask'].to(self.device)
                
                # Create target input
                tgt_input = labels.clone()
                tgt_input = torch.cat([
                    torch.full((tgt_input.size(0), 1), self.tokenizer.bos_token_id, dtype=tgt_input.dtype, device=self.device),
                    tgt_input[:, :-1]
                ], dim=1)
                
                outputs = self.model(
                    src=input_ids,
                    tgt=tgt_input,
                    src_key_padding_mask=(attention_mask == 0),
                    tgt_key_padding_mask=(target_attention_mask == 0)
                )
                
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))
                total_loss += loss.item()
        
        self.model.train()
        return total_loss / len(val_loader)
    
    def generate_answer(self, question: str, max_length: int = 100) -> str:
        """Generate answer for a given question"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")
        
        self.model.eval()
        
        # Prepare input
        source_text = f"Telugu health Q&A: {question}"
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.config.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = source_encoding['input_ids'].to(self.device)
        attention_mask = source_encoding['attention_mask'].to(self.device)
        
        # Generate response
        with torch.no_grad():
            # Start with BOS token
            generated = torch.tensor([[self.tokenizer.bos_token_id]], device=self.device)
            
            for _ in range(max_length):
                outputs = self.model(
                    src=input_ids,
                    tgt=generated,
                    src_key_padding_mask=(attention_mask == 0)
                )
                
                # Get next token
                next_token_logits = outputs[0, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)
                
                # Stop if EOS token is generated
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                generated = torch.cat([generated, next_token], dim=1)
            
            # Decode generated tokens
            generated_ids = generated[0].cpu().tolist()
            generated_text = self.tokenizer.decode(generated_ids[1:])  # Skip BOS token
            
        return generated_text
    
    def save_model(self, path: str):
        """Save model and tokenizer"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'tokenizer_model': f"telugu_health_tokenizer.model"
        }, path)
        
        # Copy tokenizer model
        import shutil
        if os.path.exists("telugu_health_tokenizer.model"):
            shutil.copy("telugu_health_tokenizer.model", f"{self.config.output_dir}/tokenizer.model")
    
    def load_model(self, path: str):
        """Load saved model and tokenizer"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load tokenizer
        tokenizer_path = f"{os.path.dirname(path)}/tokenizer.model"
        if os.path.exists(tokenizer_path):
            self.tokenizer = TeluguTokenizer()
            self.tokenizer.load_tokenizer(tokenizer_path)
        
        # Create and load model
        vocab_size = self.tokenizer.sp_model.get_piece_size() if self.tokenizer else 16000
        self.model = SimplifiedMT5Model(vocab_size=vocab_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        logger.info(f"Model loaded from {path}")

def load_dataset_from_json(file_path: str) -> List[Dict]:
    """Load dataset from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} samples from {file_path}")
    return data

def create_sample_dataset() -> List[Dict]:
    """Create a sample Telugu health dataset"""
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
            "answer": "దగ్గుకు తేనె మరియు అల్లం కలిపిన వేడిమైన నీరు త్రాగండి, ఆవిରి పీల్చుకోండి మరియు తగినంత విశ్రాంతి తీసుకోండి. రెండు వారాలకంటే ఎక్కువ కొనసాగితే వైద్యుడిని సంప్రదించండి."
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
            "answer": "బరువు తగ్గడానికి కెలోరీల తేల్చుకోండి, ఫైబర్ అధికంగా ఉన్న ఆహారాలు తీసుకోండి, రోజువారీ వ్యాయామం చేయండి మరియు తగినంత నీరు త్రాగండి. వేగవంతమైన బరువు తగ్గింపు కోసం వైద్యుడిని సంప్రదించండి."
        }
    ]
    
    return sample_data

if __name__ == "__main__":
    # Example usage
    config = TrainingConfig(
        batch_size=4,
        num_epochs=2,
        learning_rate=1e-4,
        output_dir="./mt5_telugu_health_output"
    )
    
    # Create sample dataset
    train_data = create_sample_dataset()
    
    # Initialize fine-tuner
    fine_tuner = MT5FineTuner(config)
    
    # Prepare data and train
    fine_tuner.prepare_data(train_data)
    fine_tuner.train()
    
    # Test generation
    question = "తలనొప్పికి ఏమి చేయాలి?"
    answer = fine_tuner.generate_answer(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")