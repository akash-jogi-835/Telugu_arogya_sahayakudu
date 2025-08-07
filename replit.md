# Telugu Health Q&A System

## Overview

This is a Telugu Health Q&A system built with Streamlit that simulates an MT5 fine-tuned model for providing health-related answers in Telugu. The application serves as a prototype for demonstrating Telugu medical question-answering capabilities through a web interface.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application framework
- **Interface Design**: Multi-page application with sidebar navigation
- **Caching Strategy**: Uses `@st.cache_resource` for component initialization to improve performance
- **Page Structure**: Four main sections - Q&A Interface, Dataset Upload, Model Training, and Evaluation Metrics

### Backend Architecture
- **Core Components**: 
  - `TeluguHealthQA`: Mock Q&A system that simulates MT5 model responses
  - `TeluguTextProcessor`: Text processing utilities for Telugu language
- **Language Processing**: Basic Telugu text preprocessing and validation
- **Response Generation**: Pattern matching against predefined Q&A pairs

### Data Storage Solutions
- **In-Memory Storage**: Predefined Q&A pairs stored in Python data structures
- **No Database**: Currently uses hardcoded data for simulation purposes
- **Caching**: Streamlit's built-in caching for component persistence

## Key Components

### TeluguHealthQA System
- **Purpose**: Simulates a fine-tuned MT5 model for Telugu health Q&A
- **Data Structure**: Contains 8+ predefined Telugu health Q&A pairs
- **Topics Covered**: Common health issues like headaches, fever, stomach pain, cough, diabetes, blood pressure, insomnia, and back pain
- **Response Logic**: Simple pattern matching and random selection from relevant responses

### TeluguTextProcessor
- **Text Preprocessing**: Handles Telugu Unicode text normalization
- **Language Validation**: Checks if input text contains Telugu characters
- **Health Term Mapping**: Maps Telugu health terms to English equivalents
- **Cleaning**: Removes special characters while preserving Telugu text and basic punctuation

### Streamlit Application
- **Multi-page Structure**: Organized with sidebar navigation
- **Component Caching**: Efficient resource management for Q&A system and text processor
- **User Interface**: Clean, medical-themed interface with emoji indicators

## Data Flow

1. **User Input**: User enters Telugu health question through Streamlit interface
2. **Text Processing**: Input is preprocessed using TeluguTextProcessor
3. **Question Matching**: TeluguHealthQA system matches question against predefined patterns
4. **Response Generation**: System returns appropriate Telugu health advice
5. **Display**: Answer is rendered in the Streamlit interface

## External Dependencies

### Python Packages
- **streamlit**: Web application framework
- **pandas**: Data manipulation (imported but not actively used in current implementation)
- **json**: JSON handling utilities
- **time**: Timing utilities
- **random**: Random selection for response variation
- **re**: Regular expressions for text processing

### Language Processing
- **Unicode Support**: Telugu Unicode range (U+0C00-U+0C7F) handling
- **No ML Libraries**: Currently uses rule-based approach rather than actual ML models

## Deployment Strategy

### Current Setup
- **Local Development**: Designed for local Streamlit deployment
- **No Database Dependencies**: Self-contained with hardcoded data
- **Minimal Requirements**: Only requires Python with listed dependencies

### Scalability Considerations
- **Model Integration**: Architecture prepared for actual MT5 model integration
- **Data Expansion**: Structure allows for easy addition of more Q&A pairs
- **Database Integration**: Can be extended to use external data sources
- **Multi-language Support**: Framework can be extended for other Indian languages

### Development Approach
- **Prototype Focus**: Current implementation serves as proof-of-concept
- **Modular Design**: Separate classes for Q&A logic and text processing
- **Extension Ready**: Structure supports future ML model integration and database connectivity