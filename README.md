# AadiShakthiSLM - Indic Language Summarizer

A state-of-the-art text summarization system specifically designed for Telugu and Hindi languages, implementing a hybrid neural-extractive approach with multiple model fallbacks.

## Overview

This project provides an intelligent text summarization solution for Indic languages (Telugu and Hindi) that combines advanced neural models with robust extractive methods. The system automatically detects the input language and applies language-specific processing techniques for optimal results.

## Key Features

### 🧠 **Hybrid Architecture**
- **Neural Models**: IndicBART (optimized for Indic languages) and mT5 (multilingual)
- **Extractive Fallbacks**: Sumy LexRank, LSA, and TextRank algorithms
- **Intelligent Switching**: Automatically selects the best available method

### 🔍 **Language Detection**
- Robust script-based detection for Telugu (Unicode: U+0C00-U+0C7F) and Hindi (Unicode: U+0900-U+097F)
- Confidence scoring system with pattern-based fallbacks
- Support for mixed content and transliterated text

### 📝 **Advanced Text Processing**
- Language-specific stopword filtering with comprehensive coverage
- Domain-specific term weighting (politics, economics, technology, health, education, sports)
- Intelligent sentence boundary detection for Indic scripts
- Custom tokenization for Telugu and Hindi

### 🔗 **Discourse Enhancement**
- Context-aware discourse connectors (causal, additive, contrastive, temporal, summary, emphasis)
- Intelligent sentence compression with important term preservation
- Flow enhancement for coherent summaries

### ⚡ **Robust Fallback System**
- Multiple model fallbacks ensure consistent operation
- Graceful degradation when neural models are unavailable
- Manual extractive methods as final fallback

## Technical Innovations

### 1. **Language-Specific Resources**
- **Comprehensive Stopwords**: Curated lists for Telugu and Hindi including particles, conjunctions, and common words
- **Weighted Term Importance**: Domain-specific vocabulary with importance scores (3.0 for high-importance terms like 'ప్రధానమంత్రి', 'प्रधानमंत्री')
- **Discourse Connectors**: Six categories of connectors for natural summary flow

### 2. **Intelligent Sentence Selection**
- **Diversity-based Selection**: Prevents redundancy by tracking topic coverage
- **Position and Length Weighting**: Boosts first/last sentences and optimal-length sentences
- **Similarity Filtering**: Avoids selecting sentences with >60% word overlap

### 3. **Smart Text Preprocessing**
- **Unicode Normalization**: Handles Indic sentence terminators (।)
- **Pattern-based Splitting**: Recognizes Telugu patterns like "అయ్యారు", "అయింది", "వచ్చింది"
- **Minimum Length Filtering**: Excludes fragments shorter than 15 characters

### 4. **Adaptive Summarization**
- **Length-aware Processing**: Adjusts strategy based on input and target length
- **Smart Truncation**: Cuts at sentence boundaries when possible
- **Enhancement Validation**: Checks neural enhancement effectiveness

## Model Support

### Primary Models
- **IndicBART** (`ai4bharat/IndicBART`): Specialized for Indic languages
- **mT5** (`google/mt5-base`): Strong multilingual capabilities
- **SentenceTransformer** (`all-MiniLM-L6-v2`): Semantic similarity

### Extractive Fallbacks
- **Sumy LexRank**: Graph-based sentence ranking
- **Sumy LSA**: Latent semantic analysis
- **Sumy TextRank**: PageRank for sentences

## Architecture

```
Input Text
    ↓
Language Detection (Telugu/Hindi)
    ↓
Text Preprocessing & Sentence Segmentation
    ↓
Word Frequency Analysis with Importance Weighting
    ↓
Sentence Scoring & Selection
    ↓
Neural Enhancement (if available) → Extractive Summary (fallback)
    ↓
Discourse Enhancement & Flow Improvement
    ↓
Smart Truncation & Final Summary
```

## Usage

```python
from proper_slm_summarizer import StateOfTheArtIndicSummarizer

# Initialize the summarizer
summarizer = StateOfTheArtIndicSummarizer()

# Summarize Telugu text
telugu_text = "మీ టెలుగు వచనం ఇక్కడ..."
summary, method = summarizer.summarize(telugu_text, max_length=250)

print(f"Summary: {summary}")
print(f"Method used: {method}")
```

## Installation

```bash
# Install dependencies
pip install torch transformers sentence-transformers sumy

# For GPU support (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- Sentence Transformers
- Sumy
- Regular expressions support for Unicode

## Performance

- **Languages**: Telugu, Hindi
- **Input Size**: Handles documents from single sentences to long articles
- **Output**: Configurable length (default: 250 characters)
- **Processing**: Real-time for typical news articles
- **Fallback**: 100% uptime with multiple fallback mechanisms

## File Structure

```
AadiShakthiSLM/
├── proper_slm_summarizer.py    # Main summarizer implementation
├── api/                        # API server implementations
├── demos/                      # Example scripts and demonstrations
├── tests/                      # Test scripts and validation
├── training/                   # Training scripts and utilities
├── docs/                       # Documentation
└── requirements.txt            # Dependencies
```

## License

This project is developed for research and educational purposes in Indic NLP.

## Contributing

Contributions are welcome! Please ensure all changes maintain the language-specific optimizations and fallback mechanisms.
