"""
Integrated Streamlit app that connects to the real FastAPI backend
for Telugu Health Q&A with actual ML model training and inference
"""

import streamlit as st
import requests
import json
import time
import pandas as pd
from typing import List, Dict, Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:8000"
API_TIMEOUT = 30

# Initialize session state
if 'api_available' not in st.session_state:
    st.session_state.api_available = False
if 'model_status' not in st.session_state:
    st.session_state.model_status = {}

def check_api_status():
    """Check if API server is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            st.session_state.api_available = True
            return True
    except:
        st.session_state.api_available = False
        return False

def get_model_status():
    """Get current model status from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/model-status", timeout=10)
        if response.status_code == 200:
            st.session_state.model_status = response.json()
            return st.session_state.model_status
    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        return {}

def ask_question(question: str, max_length: int = 100):
    """Send question to API and get response"""
    try:
        payload = {
            "question": question,
            "max_length": max_length
        }
        response = requests.post(
            f"{API_BASE_URL}/ask", 
            json=payload, 
            timeout=API_TIMEOUT
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code} - {response.text}"}
    
    except requests.exceptions.Timeout:
        return {"error": "Request timeout. The model might be training or processing."}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}

def train_model(dataset: List[Dict]):
    """Send training request to API"""
    try:
        payload = {"dataset": dataset}
        response = requests.post(
            f"{API_BASE_URL}/train",
            json=payload,
            timeout=120  # Training might take longer
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Training failed: {response.status_code} - {response.text}"}
    
    except requests.exceptions.Timeout:
        return {"error": "Training timeout. This might take a while for large datasets."}
    except Exception as e:
        return {"error": f"Training error: {str(e)}"}

def get_sample_questions():
    """Get sample questions from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/sample-questions", timeout=10)
        if response.status_code == 200:
            return response.json().get("sample_questions", [])
    except:
        pass
    
    # Fallback sample questions
    return [
        "తలనొప్పికి ఏమి చేయాలి?",
        "జ్వరం వచ్చినప్పుడు ఏమి చేయాలి?",
        "కడుపునొప్పికి ఏమి చేయాలి?",
        "దగ్గుకు ఏమి చేయాలి?",
        "మధుమేహం ఉన్నవారు ఏమి తినాలి?"
    ]

def main():
    st.set_page_config(
        page_title="Telugu Health Q&A System - Real ML",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Telugu Health Q&A System")
    st.markdown("### Real ML Model with FastAPI Backend")
    
    # Check API status
    api_status = check_api_status()
    model_status = get_model_status()
    
    # Display status
    col1, col2, col3 = st.columns(3)
    with col1:
        if api_status:
            st.success("🟢 API Server: Online")
        else:
            st.error("🔴 API Server: Offline")
    
    with col2:
        if model_status.get('model_loaded', False):
            st.success("🟢 Model: Loaded")
        else:
            st.warning("🟡 Model: Not Loaded")
    
    with col3:
        if st.button("🔄 Refresh Status"):
            st.rerun()
    
    if not api_status:
        st.error("""
        **API Server is not running!**
        
        To start the server, run this command in your terminal:
        ```
        python simple_fastapi_server.py
        ```
        
        The server should start on http://localhost:8000
        """)
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Q&A Interface", "Model Training", "Dataset Management", "Model Status"]
    )
    
    if page == "Q&A Interface":
        qa_interface_page()
    elif page == "Model Training":
        training_page()
    elif page == "Dataset Management":
        dataset_page()
    elif page == "Model Status":
        status_page()

def qa_interface_page():
    """Main Q&A interface"""
    st.header("Telugu Health Question & Answer")
    
    # Check if model is ready
    model_status = st.session_state.model_status
    if not model_status.get('model_loaded', False):
        st.warning("⚠️ Model is not loaded yet. Please wait for initialization or train a new model.")
        if st.button("Initialize Model"):
            with st.spinner("Initializing model with sample data..."):
                # Trigger model initialization by making a request
                sample_response = ask_question("తలనొప్పికి ఏమి చేయాలి?")
                if 'error' not in sample_response:
                    st.success("Model initialized successfully!")
                    st.rerun()
                else:
                    st.error(f"Initialization failed: {sample_response['error']}")
        return
    
    # Question input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        question = st.text_area(
            "Enter your health question in Telugu:",
            height=100,
            placeholder="ఉదాహరణ: తలనొప్పికి ఏమి చేయాలి?",
            key="main_question"
        )
        
        max_length = st.slider("Maximum answer length:", 50, 200, 100)
        
        if st.button("Get Answer", type="primary", key="get_answer"):
            if question.strip():
                with st.spinner("Processing your question..."):
                    response = ask_question(question.strip(), max_length)
                    
                    if 'error' in response:
                        st.error(f"Error: {response['error']}")
                    else:
                        st.success("Answer generated!")
                        
                        # Display results
                        st.subheader("Results")
                        st.write(f"**Question:** {response['question']}")
                        st.info(f"**Answer:** {response['answer']}")
                        
                        # Show metrics
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Confidence", f"{response['confidence']:.2f}")
                        with col_b:
                            st.metric("Processing Time", f"{response['processing_time']:.2f}s")
                        with col_c:
                            st.metric("Answer Length", len(response['answer']))
            else:
                st.error("Please enter a question in Telugu.")
    
    with col2:
        st.subheader("Sample Questions")
        sample_questions = get_sample_questions()
        
        for i, sample in enumerate(sample_questions):
            if st.button(f"📋 {sample[:25]}...", key=f"sample_{i}"):
                st.session_state.main_question = sample
                st.rerun()

def training_page():
    """Model training interface"""
    st.header("Model Training")
    
    st.markdown("""
    Train the Telugu Health Q&A model with your own dataset.
    Upload or create a dataset with Telugu question-answer pairs.
    """)
    
    # Training options
    training_method = st.radio(
        "Choose training method:",
        ["Upload JSON Dataset", "Create Dataset Manually", "Use Extended Sample Data"]
    )
    
    dataset = []
    
    if training_method == "Upload JSON Dataset":
        uploaded_file = st.file_uploader(
            "Upload JSON file with Q&A pairs",
            type=['json'],
            help="Format: [{'question': 'Telugu question', 'answer': 'Telugu answer'}, ...]"
        )
        
        if uploaded_file:
            try:
                content = uploaded_file.read()
                dataset = json.loads(content.decode('utf-8'))
                
                st.success(f"Loaded {len(dataset)} Q&A pairs")
                
                # Show preview
                if dataset:
                    st.subheader("Dataset Preview")
                    df = pd.DataFrame(dataset[:5])  # Show first 5
                    st.dataframe(df, use_container_width=True)
                
            except json.JSONDecodeError:
                st.error("Invalid JSON format")
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    elif training_method == "Create Dataset Manually":
        st.subheader("Create Training Dataset")
        
        # Initialize session state for manual dataset
        if 'manual_dataset' not in st.session_state:
            st.session_state.manual_dataset = []
        
        # Add new Q&A pair
        with st.form("add_qa_pair"):
            question = st.text_area("Question (Telugu):", height=80)
            answer = st.text_area("Answer (Telugu):", height=100)
            
            if st.form_submit_button("Add Q&A Pair"):
                if question.strip() and answer.strip():
                    st.session_state.manual_dataset.append({
                        "question": question.strip(),
                        "answer": answer.strip()
                    })
                    st.success("Q&A pair added!")
                    st.rerun()
                else:
                    st.error("Please fill both question and answer fields")
        
        # Show current dataset
        if st.session_state.manual_dataset:
            st.subheader(f"Current Dataset ({len(st.session_state.manual_dataset)} pairs)")
            df = pd.DataFrame(st.session_state.manual_dataset)
            st.dataframe(df, use_container_width=True)
            
            if st.button("Clear Dataset"):
                st.session_state.manual_dataset = []
                st.rerun()
            
            dataset = st.session_state.manual_dataset
    
    elif training_method == "Use Extended Sample Data":
        # Extended sample dataset
        extended_sample = [
            {"question": "తలనొప్పికి ఏమి చేయాలి?", "answer": "తలనొప్పికి విశ్రాంతి తీసుకోండి, నీరు ఎక్కువగా త్రాగండి మరియు అవసరమైతే పారాసిటమాల్ తీసుకోండి."},
            {"question": "జ్వరం వచ్చినప్పుడు ఏమి చేయాలి?", "answer": "జ్వరం వచ్చినప్పుడు విశ్రాంతి తీసుకోండి, ద్రవాలు ఎక్కువగా త్రాగండి మరియు చల్లని వస్త్రంతో శరీరాన్ని తుడుచుకోండి."},
            {"question": "కడుపునొప్పికి ఏమి చేయాలి?", "answer": "కడుపునొప్పికి అల్లం టీ త్రాగండి, తేలికపాటి ఆహారం తీసుకోండి మరియు వేడిమిని కడుపుపై పెట్టండి."},
            {"question": "దగ్గుకు ఏమి చేయాలి?", "answer": "దగ్గుకు తేనె మరియు అల్లం కలిపిన వేడిమైన నీరు త్రాగండి, ఆవిరి పీల్చుకోండి మరియు తగినంత విశ్రాంతి తీసుకోండి."},
            {"question": "మధుమేహం ఉన్నవారు ఏమి తినాలి?", "answer": "మధుమేహం ఉన్నవారు తక్కువ గ్లైసెమిక్ ఇండెక్స్ ఉన్న ఆహారాలు తీసుకోవాలి. కూరగాయలు, ధాన్యాలు, మరియు ప్రోటీన్ ఆహారాలు మంచివి."},
            {"question": "రక్తపోటు ఎక్కువగా ఉంటే ఏమి చేయాలి?", "answer": "రక్తపోటు ఎక్కువగా ఉంటే ఉప్పు తక్కువగా తీసుకోండి, నిత్యం వ్యాయామం చేయండి, మరియు ఒత్తిడిని తగ్గించండి."},
            {"question": "నిద్రలేకపోవడానికి ఏమి చేయాలి?", "answer": "నిద్రలేకపోవడానికి రోజువారీ వ్యాయామం చేయండి, కెఫీన్ తగ్గించండి మరియు నిద్రకు ముందు రిలాక్సేషన్ టెక్నిక్స్ చేయండి."},
            {"question": "వెన్నునొప్పికి ఏమి చేయాలి?", "answer": "వెన్నునొప్పికి వేడిమిని లేదా చల్లదనాన్ని ప్రయోగించండి, తేలికపాటి వ్యాయామాలు చేయండి మరియు సరైన భంగిమలో కూర్చోండి."},
            {"question": "డయాబెటిస్ ఎలా నియంత్రించాలి?", "answer": "డయాబెటిస్ నియంత్రణకు సమతుల్య ఆహారం, నిత్య వ్యాయామం, మందుల సక్రమ సేవన అవసరం."},
            {"question": "అధిక బరువు తగ్గడానికి ఏమి చేయాలి?", "answer": "బరువు తగ్గడానికి కెలోరీలను తగ్గించండి, ఫైబర్ అధికంగా ఉన్న ఆహారాలు తీసుకోండి, రోజువారీ వ్యాయామం చేయండి."}
        ]
        
        st.subheader("Extended Sample Dataset")
        df = pd.DataFrame(extended_sample)
        st.dataframe(df, use_container_width=True)
        dataset = extended_sample
    
    # Training button
    if dataset and len(dataset) >= 3:
        st.subheader("Start Training")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Dataset Size", len(dataset))
        with col2:
            avg_q_len = sum(len(item['question']) for item in dataset) // len(dataset)
            st.metric("Avg Question Length", f"{avg_q_len} chars")
        
        if st.button("🚀 Start Training", type="primary"):
            with st.spinner("Training model... This may take a few minutes."):
                result = train_model(dataset)
                
                if 'error' in result:
                    st.error(f"Training failed: {result['error']}")
                else:
                    st.success("🎉 Training completed successfully!")
                    st.json(result)
                    
                    # Refresh model status
                    get_model_status()
                    st.rerun()
    
    elif dataset:
        st.warning("Dataset must contain at least 3 Q&A pairs for training.")
    else:
        st.info("Please create or upload a dataset to start training.")

def dataset_page():
    """Dataset management page"""
    st.header("Dataset Management")
    
    st.markdown("Manage your Telugu Health Q&A datasets")
    
    # Sample dataset generator
    st.subheader("Generate Sample Dataset")
    
    if st.button("Download Sample Dataset JSON"):
        sample_data = [
            {"question": "తలనొప్పికి ఏమి చేయాలి?", "answer": "తలనొప్పికి విశ్రాంతి తీసుకోండి మరియు నీరు త్రాగండి."},
            {"question": "జ్వరం వచ్చినప్పుడు ఏమి చేయాలి?", "answer": "జ్వరం వచ్చినప్పుడు విశ్రాంతి తీసుకోండి మరియు ద్రవాలు త్రాగండి."},
            {"question": "కడుపునొప్పికి ఏమి చేయాలి?", "answer": "కడుపునొప్పికి అల్లం టీ త్రాగండి మరియు విశ్రాంతి తీసుకోండి."}
        ]
        
        st.download_button(
            label="📄 Download sample_dataset.json",
            data=json.dumps(sample_data, ensure_ascii=False, indent=2),
            file_name="sample_telugu_health_qa.json",
            mime="application/json"
        )
    
    # Dataset validation
    st.subheader("Dataset Validation")
    
    uploaded_file = st.file_uploader(
        "Upload dataset for validation",
        type=['json'],
        key="validation_upload"
    )
    
    if uploaded_file:
        try:
            content = uploaded_file.read()
            data = json.loads(content.decode('utf-8'))
            
            # Validate dataset
            errors = []
            valid_pairs = 0
            
            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    errors.append(f"Item {i}: Not a dictionary")
                    continue
                
                if 'question' not in item:
                    errors.append(f"Item {i}: Missing 'question' field")
                elif len(item['question'].strip()) < 5:
                    errors.append(f"Item {i}: Question too short")
                
                if 'answer' not in item:
                    errors.append(f"Item {i}: Missing 'answer' field")
                elif len(item['answer'].strip()) < 10:
                    errors.append(f"Item {i}: Answer too short")
                
                if 'question' in item and 'answer' in item and len(item['question'].strip()) >= 5 and len(item['answer'].strip()) >= 10:
                    valid_pairs += 1
            
            # Display validation results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Items", len(data))
            with col2:
                st.metric("Valid Pairs", valid_pairs)
            with col3:
                st.metric("Errors", len(errors))
            
            if errors:
                st.error("Validation Errors:")
                for error in errors[:10]:  # Show first 10 errors
                    st.write(f"• {error}")
                if len(errors) > 10:
                    st.write(f"... and {len(errors) - 10} more errors")
            else:
                st.success("✅ Dataset is valid!")
            
            # Show sample data
            if data:
                st.subheader("Dataset Preview")
                df = pd.DataFrame(data[:5])
                st.dataframe(df, use_container_width=True)
        
        except json.JSONDecodeError:
            st.error("Invalid JSON format")
        except Exception as e:
            st.error(f"Error processing file: {e}")

def status_page():
    """Model status and information page"""
    st.header("Model Status")
    
    # Get fresh status
    model_status = get_model_status()
    
    if model_status:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Current Status")
            if model_status.get('model_loaded', False):
                st.success("🟢 Model is loaded and ready")
            else:
                st.warning("🟡 Model is not loaded")
            
            st.write(f"**Last Updated:** {model_status.get('last_updated', 'Unknown')}")
            st.write(f"**Model Path:** {model_status.get('model_path', 'Not available')}")
        
        with col2:
            st.subheader("Actions")
            if st.button("🔄 Refresh Status"):
                get_model_status()
                st.rerun()
            
            if st.button("🔧 Reinitialize Model"):
                with st.spinner("Reinitializing model..."):
                    # Trigger reinitialization
                    sample_response = ask_question("తలనొప్పికి ఏమి చేయాలి?")
                    if 'error' not in sample_response:
                        st.success("Model reinitialized successfully!")
                        st.rerun()
                    else:
                        st.error(f"Reinitialization failed: {sample_response['error']}")
    
    # API Information
    st.subheader("API Information")
    st.write(f"**API Base URL:** {API_BASE_URL}")
    
    # Test API endpoints
    st.subheader("Test API Endpoints")
    
    endpoints = [
        ("/health", "Health Check"),
        ("/model-status", "Model Status"),
        ("/sample-questions", "Sample Questions")
    ]
    
    for endpoint, description in endpoints:
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write(f"**{description}**")
            st.code(f"GET {API_BASE_URL}{endpoint}")
        
        with col2:
            if st.button(f"Test", key=f"test_{endpoint}"):
                try:
                    response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=5)
                    if response.status_code == 200:
                        st.success("✅")
                    else:
                        st.error("❌")
                except:
                    st.error("❌")
        
        with col3:
            st.write("")  # Spacer

if __name__ == "__main__":
    main()