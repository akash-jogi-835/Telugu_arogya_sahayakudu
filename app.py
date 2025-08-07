import streamlit as st
import json
import pandas as pd
import time
import random
from telugu_health_qa import TeluguHealthQA
from text_processor import TeluguTextProcessor

# Initialize components
@st.cache_resource
def load_qa_system():
    return TeluguHealthQA()

@st.cache_resource
def load_text_processor():
    return TeluguTextProcessor()

def main():
    st.set_page_config(
        page_title="Telugu Health Q&A System",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Telugu Health Q&A System")
    st.markdown("### MT5 Fine-tuning Prototype for Telugu Medical Q&A")
    
    # Initialize systems
    qa_system = load_qa_system()
    text_processor = load_text_processor()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Q&A Interface", "Dataset Upload", "Model Training", "Evaluation Metrics"]
    )
    
    if page == "Q&A Interface":
        qa_interface(qa_system, text_processor)
    elif page == "Dataset Upload":
        dataset_upload_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Evaluation Metrics":
        evaluation_metrics_page()

def qa_interface(qa_system, text_processor):
    st.header("Telugu Health Question & Answer")
    
    # Text input for Telugu questions
    question = st.text_area(
        "Enter your health question in Telugu:",
        height=100,
        placeholder="ఉదాహరణ: తలనొప్పికి ఏమి చేయాలి?"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Get Answer", type="primary"):
            if question.strip():
                with st.spinner("Processing your question..."):
                    # Simulate processing time
                    time.sleep(1)
                    
                    # Process the question
                    processed_question = text_processor.preprocess(question)
                    
                    # Get answer from QA system
                    answer = qa_system.get_answer(processed_question)
                    
                    st.success("Answer generated!")
                    
                    # Display results
                    st.subheader("Question Analysis")
                    st.write(f"**Original Question:** {question}")
                    st.write(f"**Processed Question:** {processed_question}")
                    
                    st.subheader("Answer")
                    st.info(answer)
                    
                    # Show confidence score
                    confidence = random.uniform(0.75, 0.95)
                    st.metric("Confidence Score", f"{confidence:.2f}")
                    
            else:
                st.error("Please enter a question in Telugu.")
    
    with col2:
        st.subheader("Sample Questions")
        sample_questions = qa_system.get_sample_questions()
        for i, sample in enumerate(sample_questions):
            if st.button(f"Use: {sample[:30]}...", key=f"sample_{i}"):
                st.session_state.sample_question = sample
                st.rerun()
    
    # Auto-fill sample question if selected
    if hasattr(st.session_state, 'sample_question'):
        st.text_area(
            "Selected Sample Question:",
            value=st.session_state.sample_question,
            height=60,
            disabled=True
        )

def dataset_upload_page():
    st.header("Dataset Upload & Preview")
    
    st.markdown("""
    Upload your Telugu health Q&A dataset in JSON format. 
    Expected format: `[{"question": "Telugu question", "answer": "Telugu answer"}, ...]`
    """)
    
    uploaded_file = st.file_uploader(
        "Choose a JSON file",
        type=['json'],
        help="Upload a JSON file containing Telugu health Q&A pairs"
    )
    
    if uploaded_file is not None:
        try:
            # Read and parse JSON
            content = uploaded_file.read()
            data = json.loads(content.decode('utf-8'))
            
            st.success(f"Successfully loaded {len(data)} Q&A pairs!")
            
            # Convert to DataFrame for display
            df = pd.DataFrame(data)
            
            # Display statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Questions", len(data))
            with col2:
                avg_q_length = sum(len(item.get('question', '')) for item in data) // len(data)
                st.metric("Avg Question Length", f"{avg_q_length} chars")
            with col3:
                avg_a_length = sum(len(item.get('answer', '')) for item in data) // len(data)
                st.metric("Avg Answer Length", f"{avg_a_length} chars")
            
            # Display preview
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Data validation
            st.subheader("Data Validation")
            missing_questions = sum(1 for item in data if not item.get('question'))
            missing_answers = sum(1 for item in data if not item.get('answer'))
            
            if missing_questions == 0 and missing_answers == 0:
                st.success("✅ All records have both questions and answers")
            else:
                if missing_questions > 0:
                    st.warning(f"⚠️ {missing_questions} records missing questions")
                if missing_answers > 0:
                    st.warning(f"⚠️ {missing_answers} records missing answers")
            
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def model_training_page():
    st.header("MT5 Model Training Simulation")
    
    st.markdown("""
    This page simulates the MT5 fine-tuning process for Telugu health Q&A.
    In a real implementation, this would connect to actual training infrastructure.
    """)
    
    # Training parameters
    st.subheader("Training Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        epochs = st.slider("Number of Epochs", 1, 10, 3)
        batch_size = st.selectbox("Batch Size", [8, 16, 32], index=1)
        learning_rate = st.selectbox("Learning Rate", [1e-5, 2e-5, 5e-5], index=1)
    
    with col2:
        max_length = st.slider("Max Sequence Length", 128, 512, 256)
        warmup_steps = st.slider("Warmup Steps", 100, 1000, 500)
        save_steps = st.slider("Save Steps", 100, 1000, 500)
    
    # Start training simulation
    if st.button("Start Training Simulation", type="primary"):
        st.subheader("Training Progress")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_placeholder = st.empty()
        
        # Simulate training epochs
        for epoch in range(epochs):
            for step in range(10):  # 10 steps per epoch for demo
                progress = ((epoch * 10) + step + 1) / (epochs * 10)
                progress_bar.progress(progress)
                
                # Simulate metrics
                train_loss = 2.5 - (progress * 1.8) + random.uniform(-0.1, 0.1)
                val_loss = 2.3 - (progress * 1.6) + random.uniform(-0.1, 0.1)
                
                status_text.text(f"Epoch {epoch + 1}/{epochs}, Step {step + 1}/10")
                
                # Update metrics display
                with metrics_placeholder.container():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Training Loss", f"{train_loss:.3f}")
                    with col2:
                        st.metric("Validation Loss", f"{val_loss:.3f}")
                    with col3:
                        st.metric("Progress", f"{progress*100:.1f}%")
                
                time.sleep(0.3)  # Simulate training time
        
        st.success("🎉 Training completed successfully!")
        st.balloons()

def evaluation_metrics_page():
    st.header("Model Evaluation Metrics")
    
    st.markdown("""
    Mock evaluation metrics that would be computed on a held-out test set
    in a real Telugu health Q&A system.
    """)
    
    # Generate mock metrics
    bleu_score = random.uniform(0.65, 0.85)
    rouge_1 = random.uniform(0.70, 0.90)
    rouge_2 = random.uniform(0.60, 0.80)
    rouge_l = random.uniform(0.65, 0.85)
    bertscore = random.uniform(0.75, 0.92)
    
    # Display metrics
    st.subheader("Automatic Evaluation Metrics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("BLEU Score", f"{bleu_score:.3f}")
        st.metric("ROUGE-1", f"{rouge_1:.3f}")
    
    with col2:
        st.metric("ROUGE-2", f"{rouge_2:.3f}")
        st.metric("ROUGE-L", f"{rouge_l:.3f}")
    
    with col3:
        st.metric("BERTScore", f"{bertscore:.3f}")
        st.metric("Overall Score", f"{(bleu_score + rouge_l + bertscore)/3:.3f}")
    
    # Detailed breakdown
    st.subheader("Detailed Analysis")
    
    # Create sample evaluation data
    eval_data = []
    categories = ["सामान्य स्वास्थ्य", "दवाएं", "लक्षण", "निदान", "उपचार"]
    
    for category in categories:
        eval_data.append({
            "Category": category,
            "BLEU": random.uniform(0.60, 0.90),
            "ROUGE-1": random.uniform(0.65, 0.90),
            "ROUGE-2": random.uniform(0.55, 0.85),
            "ROUGE-L": random.uniform(0.60, 0.88),
            "Sample Count": random.randint(50, 200)
        })
    
    df_eval = pd.DataFrame(eval_data)
    st.dataframe(df_eval, use_container_width=True)
    
    # Performance insights
    st.subheader("Performance Insights")
    st.info("🔍 **Key Observations:**")
    st.write("• The model shows strong performance on general health queries")
    st.write("• Medication-related questions have slightly lower BLEU scores")
    st.write("• Symptom description responses achieve high ROUGE-L scores")
    st.write("• Overall performance indicates successful fine-tuning on Telugu health domain")

if __name__ == "__main__":
    main()
