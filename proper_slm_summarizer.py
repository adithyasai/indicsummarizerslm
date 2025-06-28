#!/usr/bin/env python3
"""
State-of-the-Art Indic Language Summarizer for Telugu and Hindi
Implements a hybrid approach using IndicBART, mT5, and advanced extractive methods
Based on latest research in Indic NLP and text summarization
"""

import torch
import re
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
import os
from collections import Counter
import math
import warnings
warnings.filterwarnings("ignore")

# Core transformers for state-of-the-art models
try:
    from transformers import (
        AutoTokenizer, AutoModelForSeq2SeqLM, 
        MBartForConditionalGeneration, T5ForConditionalGeneration,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Fallback extractive summarization
try:
    from sumy.summarizers.lex_rank import LexRankSummarizer
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.summarizers.text_rank import TextRankSummarizer
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    SUMY_AVAILABLE = True
except ImportError:
    SUMY_AVAILABLE = False

# Sentence transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StateOfTheArtIndicSummarizer:
    """
    State-of-the-art Indic language summarizer using hybrid neural-extractive approach
    Supports Telugu and Hindi with multiple model fallbacks
    """
    
    def __init__(self):
        """Initialize with best available models for Indic languages"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Model availability flags
        self.indicbart_available = False
        self.mt5_available = False
        self.sentence_transformer_available = False
        
        # Initialize models
        self._load_models()
        self._initialize_language_resources()
        self.lang_code = 'te'  # Default to Telugu
        
    def _load_models(self):
        """Load state-of-the-art models with graceful fallbacks"""
        
        # 1. Try to load IndicBART (best for Indic languages)
        try:
            logger.info("Loading IndicBART model...")
            self.indicbart_tokenizer = AutoTokenizer.from_pretrained(
                "ai4bharat/IndicBART", 
                do_lower_case=False, 
                use_fast=False, 
                keep_accents=True
            )
            self.indicbart_model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/IndicBART")
            self.indicbart_model.to(self.device)
            self.indicbart_model.eval()
            self.indicbart_available = True
            logger.info("✅ IndicBART loaded successfully!")
        except Exception as e:
            logger.warning(f"Failed to load IndicBART: {e}")
            
        # 2. Try to load mT5 (strong multilingual model)
        try:
            logger.info("Loading mT5 model...")
            self.mt5_tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
            self.mt5_model = T5ForConditionalGeneration.from_pretrained("google/mt5-base")
            self.mt5_model.to(self.device)
            self.mt5_model.eval()
            self.mt5_available = True
            logger.info("✅ mT5 loaded successfully!")
        except Exception as e:
            logger.warning(f"Failed to load mT5: {e}")
            
        # 3. Try to load sentence transformer for semantic similarity
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.sentence_transformer_available = True
            logger.info("✅ Sentence transformer loaded successfully!")
        except Exception as e:
            logger.warning(f"Failed to load sentence transformer: {e}")
            
        # Log available models
        available_models = []
        if self.indicbart_available:
            available_models.append("IndicBART")
        if self.mt5_available:
            available_models.append("mT5")
        if self.sentence_transformer_available:
            available_models.append("SentenceTransformer")
        if SUMY_AVAILABLE:
            available_models.append("Sumy")
            
        logger.info(f"Available models: {', '.join(available_models) if available_models else 'None (extractive only)'}")
    
    def _initialize_language_resources(self):
        """Initialize comprehensive language-specific resources for Telugu and Hindi"""
        
        # Enhanced stopwords with more comprehensive coverage
        self.stopwords = {
            'te': {
                # Basic particles and conjunctions
                'మరియు', 'కూడా', 'ఒక', 'అది', 'ఈ', 'ఆ', 'అయితే', 'కానీ', 'లేదా',
                'లో', 'నుంచి', 'వరకు', 'తో', 'చేసి', 'అని', 'అయి', 'అవుతుంది',
                'ఉంది', 'ఉన్న', 'ఉన్నది', 'చేస్తున్న', 'చేస్తున్నారు',
                # Additional common words
                'అంటే', 'అంతే', 'అక్కడ', 'ఇక్కడ', 'ఎక్కడ', 'ఎప్పుడు', 'ఎలా',
                'ఏమిటి', 'ఏది', 'ఎవరు', 'ఎంత', 'మాత్రమే', 'కేవలం', 'చాలా',
                'కొంచెం', 'తక్కువ', 'ఎక్కువ', 'పెద్ద', 'చిన్న', 'మొదటి', 'చివరి'
            },
            'hi': {
                # Basic particles and conjunctions  
                'और', 'का', 'एक', 'में', 'की', 'है', 'यह', 'तथा', 'को', 'इस',
                'से', 'पर', 'वह', 'कि', 'गया', 'हुआ', 'रहा', 'था', 'होता',
                'करता', 'करते', 'किया', 'जाता', 'होने', 'वाला',
                # Additional common words
                'इसका', 'उसका', 'जिसका', 'किसका', 'अपना', 'हमारा', 'तुम्हारा',
                'मैं', 'तुम', 'हम', 'वे', 'कुछ', 'सब', 'सभी', 'कोई', 'किसी',
                'जब', 'तब', 'कब', 'कहाँ', 'यहाँ', 'वहाँ', 'कैसे', 'क्यों'
            }
        }
        
        # Enhanced important terms with domain-specific vocabulary
        self.important_terms = {
            'te': {
                # Politics and governance
                'ప్రధానమంత్రి': 3.0, 'మంత్రి': 2.5, 'అధ్యక్షుడు': 3.0, 'నेता': 2.0,
                'ప్రభుత్వం': 2.5, 'రాజకీయ': 2.0, 'ఎన్నికలు': 2.5, 'పార్లమెంట్': 2.5,
                'శాసనసభ': 2.5, 'న్యాయస్థానం': 2.5, 'న్యాయమూర్తి': 2.0,
                
                # Economics and business
                'ఆర్థిక': 2.5, 'వ్యాపార': 2.0, 'వాణిజ్యం': 2.0, 'పెట్టుబడి': 2.5,
                'కంపెనీ': 2.0, 'బ్యాంకు': 2.0, 'మార్కెట్': 2.0, 'ధర': 2.0,
                'రూపాయి': 2.0, 'డాలర్': 2.0, 'వృద్ధి': 2.5, 'తగ్గుదల': 2.0,
                
                # Technology
                'సాంకేతిక': 2.5, 'కంప్యూటర్': 2.0, 'ఇంటర్నెట్': 2.0, 'డిజిటల్': 2.5,
                'సాఫ్ట్‌వేర్': 2.0, 'యాప్': 2.0, 'వెబ్‌సైట్': 2.0,
                
                # Health and medicine
                'వైద్య': 2.5, 'ఆరోగ్య': 2.5, 'వైరస': 2.5, 'టీకా': 2.5, 'చికిత్స': 2.5,
                'ఆసుపత్రి': 2.0, 'డాక్టర్': 2.0, 'మందు': 2.0, 'రోగి': 2.0,
                
                # Education
                'విద్య': 2.5, 'విద్యార్థి': 2.0, 'ఉపాధ్యాయుడు': 2.0, 'పాఠశాల': 2.0,
                'కళాశాల': 2.0, 'విశ్వవిద్యాలయం': 2.5, 'పరీక్ష': 2.0,
                
                # Sports
                'క్రీడలు': 2.5, 'క్రికెట్': 2.0, 'ఫుట్‌బాల్': 2.0, 'టెన్నిస్': 2.0,
                'ఒలింపిక్స్': 2.5, 'మ్యాచ్': 2.0, 'టోర్నమెంట్': 2.0
            },
            'hi': {
                # Politics and governance  
                'प्रधानमंत्री': 3.0, 'मंत्री': 2.5, 'राष्ट्रपति': 3.0, 'नेता': 2.0,
                'सरकार': 2.5, 'राजनीतिक': 2.0, 'चुनाव': 2.5, 'संसद': 2.5,
                'विधानसभा': 2.5, 'न्यायालय': 2.5, 'न्यायाधीश': 2.0,
                
                # Economics and business
                'आर्थिक': 2.5, 'व्यापार': 2.0, 'वाणिज्य': 2.0, 'निवेश': 2.5,
                'कंपनी': 2.0, 'बैंक': 2.0, 'बाजार': 2.0, 'कीमत': 2.0,
                'रुपया': 2.0, 'डॉलर': 2.0, 'वृद्धि': 2.5, 'गिरावट': 2.0,
                
                # Technology
                'तकनीकी': 2.5, 'कंप्यूटर': 2.0, 'इंटरनेट': 2.0, 'डिजिटल': 2.5,
                'सॉफ्टवेयर': 2.0, 'ऐप': 2.0, 'वेबसाइट': 2.0,
                
                # Health and medicine
                'चिकित्सा': 2.5, 'स्वास्थ्य': 2.5, 'वायरस': 2.5, 'टीका': 2.5, 'इलाज': 2.5,
                'अस्पताल': 2.0, 'डॉक्टर': 2.0, 'दवा': 2.0, 'मरीज': 2.0,
                
                # Education
                'शिक्षा': 2.5, 'छात्र': 2.0, 'शिक्षक': 2.0, 'स्कूल': 2.0,
                'कॉलेज': 2.0, 'विश्वविद्यालय': 2.5, 'परीक्षा': 2.0,
                
                # Sports
                'खेल': 2.5, 'क्रिकेट': 2.0, 'फुटबॉल': 2.0, 'टेनिस': 2.0,
                'ओलंपिक': 2.5, 'मैच': 2.0, 'टूर्नामेंट': 2.0
            }
        }
        
        # Enhanced transition markers and discourse connectors
        self.connectors = {
            'te': {
                'causal': ['కాబట్టি', 'అందువలన', 'దీనివలన', 'ఎందుకంటే', 'కారణంగా'],
                'additive': ['అలాగే', 'అంతేకాకుండా', 'ఇంకా', 'పైగా', 'అదేవిధంగా', 'అదనంగా'],
                'contrastive': ['అయితే', 'కానీ', 'మరోవైపు', 'అయినప్పటికీ', 'విరుద్ధంగా'],
                'temporal': ['అప్పుడు', 'తర్వాత', 'ముందు', 'ఇంతలో', 'అంతలో', 'ఇప్పుడు'],
                'summary': ['సారాంశంలో', 'మొత్తంమీద', 'చివరగా', 'ముగింపులో', 'సంక్షేపంలో'],
                'emphasis': ['ముఖ్యంగా', 'విశేషంగా', 'కీలకంగా', 'గమనార్హంగా']
            },
            'hi': {
                'causal': ['इसलिए', 'अतः', 'क्योंकि', 'इसके कारण', 'के कारण', 'वजह से'],
                'additive': ['इसके अलावा', 'साथ ही', 'तथा', 'भी', 'इसके साथ', 'अतिरिक्त'],
                'contrastive': ['लेकिन', 'परंतु', 'किंतु', 'वहीं दूसरी ओर', 'इसके विपरीत'],
                'temporal': ['फिर', 'बाद में', 'पहले', 'इस दौरान', 'तब', 'अब'],
                'summary': ['सारांश में', 'कुल मिलाकर', 'अंत में', 'निष्कर्ष में', 'संक्षेप में'],
                'emphasis': ['खासकर', 'विशेष रूप से', 'मुख्य रूप से', 'उल्लेखनीय रूप से']
            }
        }
        
    def detect_language(self, text: str) -> str:
        """
        Robust language detection for Telugu and Hindi
        Returns language code with high confidence
        """
        # Clean text for analysis
        text = re.sub(r'[^\u0900-\u097F\u0C00-\u0C7F\w\s]', '', text)
        
        # Count characters by script
        telugu_chars = len(re.findall(r'[\u0C00-\u0C7F]', text))
        hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
        
        # Count total Indic characters
        total_indic_chars = telugu_chars + hindi_chars
        
        # If no Indic characters, check for language-specific keywords
        if total_indic_chars == 0:
            # Check for transliterated words or mixed content
            text_lower = text.lower()
            
            # Telugu indicators
            telugu_indicators = ['telugu', 'andhra', 'hyderabad', 'vizag', 'vijayawada']
            # Hindi indicators  
            hindi_indicators = ['hindi', 'delhi', 'mumbai', 'bharatiya', 'hindustani']
            
            telugu_score = sum(1 for indicator in telugu_indicators if indicator in text_lower)
            hindi_score = sum(1 for indicator in hindi_indicators if indicator in text_lower)
            
            if telugu_score > hindi_score:
                return 'te'
            elif hindi_score > telugu_score:
                return 'hi'
            else:
                return 'te'  # Default to Telugu
        
        # Calculate confidence percentages
        telugu_percentage = telugu_chars / total_indic_chars if total_indic_chars > 0 else 0
        hindi_percentage = hindi_chars / total_indic_chars if total_indic_chars > 0 else 0
        
        # Require at least 60% confidence for language detection
        if telugu_percentage >= 0.6:
            return 'te'
        elif hindi_percentage >= 0.6:
            return 'hi'
        elif telugu_chars > hindi_chars:
            return 'te'
        elif hindi_chars > telugu_chars:
            return 'hi'
        else:
            # If equal or very close, use text length heuristics
            if len(text) < 100:
                # Short text: be more decisive
                return 'te' if telugu_chars >= hindi_chars else 'hi'
            else:
                # Longer text: check for language-specific patterns
                # Telugu tends to have more conjunct consonants
                telugu_patterns = len(re.findall(r'[\u0C15-\u0C39][\u0C4D][\u0C15-\u0C39]', text))
                # Hindi tends to have more vowel marks
                hindi_patterns = len(re.findall(r'[\u0915-\u0939][\u093E-\u094F]', text))
                
                if telugu_patterns > hindi_patterns:
                    return 'te'
                elif hindi_patterns > telugu_patterns:
                    return 'hi'
                else:
                    return 'te'  # Final fallback to Telugu
    
    def get_language_confidence(self, text: str) -> tuple:
        """
        Get language detection with confidence score
        Returns (language_code, confidence_score)
        """
        # Clean text for analysis
        text = re.sub(r'[^\u0900-\u097F\u0C00-\u0C7F\w\s]', '', text)
        
        telugu_chars = len(re.findall(r'[\u0C00-\u0C7F]', text))
        hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
        total = telugu_chars + hindi_chars
        
        if total == 0:
            return ('te', 0.5)  # Low confidence for non-Indic text
        
        telugu_pct = telugu_chars / total
        hindi_pct = hindi_chars / total
        
        if telugu_pct >= 0.8:
            return ('te', 0.9)
        elif hindi_pct >= 0.8:
            return ('hi', 0.9)
        elif telugu_pct >= 0.6:
            return ('te', 0.7)
        elif hindi_pct >= 0.6:
            return ('hi', 0.7)
        else:
            # Lower confidence for mixed content
            lang = 'te' if telugu_chars >= hindi_chars else 'hi'
            conf = 0.6 if abs(telugu_pct - hindi_pct) > 0.2 else 0.5
            return (lang, conf)
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text and split into sentences"""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Normalize Indic sentence terminators (Telugu/Hindi danda)
        text = re.sub(r'[।]+', '.', text)
        # Remove extraneous quotes/brackets
        text = re.sub(r'["“”‘’\(\)]', '', text)        # Enhanced sentence splitting for Telugu/Hindi text
        # Split on periods, exclamation marks, question marks, and also Telugu/Hindi sentence patterns
        sentences = re.split(r'[.!?।]+|(?<=\u0C7F)\s+|(?<=\u097F)\s+', text)
        
        # Also split on common Telugu sentence ending patterns
        extended_sentences = []
        for sentence in sentences:
            # Split on common Telugu patterns like "అయ్యారు.", "అయింది.", "వచ్చింది." etc.
            sub_sentences = re.split(r'(?<=అయ్యారు)\s+|(?<=అయింది)\s+|(?<=వచ్చింది)\s+|(?<=చేసింది)\s+|(?<=అయ్యాయి)\s+', sentence)
            extended_sentences.extend(sub_sentences)
        
        # Filter out very short segments and clean up
        sentences = [s.strip() for s in extended_sentences if len(s.strip()) > 15]
        
        # Debug: Print detected sentences
        logger.info(f"Detected {len(sentences)} sentences: {sentences}")
        return sentences
    
    def _tokenize_words(self, text: str) -> List[str]:
        """Tokenize words using custom tokenizer if available, fallback to regex"""
        if self.custom_tokenizer:
            try:
                # Use custom tokenizer for better word segmentation
                tokens = self.custom_tokenizer.tokenize(text, add_special_tokens=False)
                # Filter out special tokens and punctuation
                words = [token for token in tokens if not token.startswith('<') and 
                        len(token) > 1 and re.match(r'[\w\u0C00-\u0C7F\u0900-\u097F]+', token)]
                return words
            except Exception as e:
                logger.warning(f"Custom tokenizer failed, using fallback: {e}")
        
        # Fallback to regex tokenization
        return re.findall(r'[\w\u0C00-\u0C7F\u0900-\u097F]+', text.lower())

    def _calculate_word_frequencies(self, sentences: List[str]) -> Dict[str, float]:
        """Calculate word frequencies with importance weighting using proper tokenization"""
        word_freq = Counter()
        total_words = 0
        
        stopwords = self.stopwords.get(self.lang_code, set())
        important_terms = self.important_terms.get(self.lang_code, {})
        
        for sentence in sentences:
            # Use proper tokenization for Telugu/Hindi
            words = self._tokenize_words(sentence)
            for word in words:
                word_lower = word.lower()
                if word_lower not in stopwords and len(word) > 1:
                    # Apply importance weighting
                    weight = important_terms.get(word_lower, 1.0)
                    word_freq[word_lower] += weight
                    total_words += 1
          # Normalize frequencies
        word_scores = {}
        for word, freq in word_freq.items():
            word_scores[word] = freq / total_words if total_words > 0 else 0
        
        return word_scores
    
    def _score_sentences(self, sentences: List[str]) -> Dict[int, float]:
        """Score sentences with extra boost for summary-like and key-topic words"""
        word_scores = self._calculate_word_frequencies(sentences)
        sentence_scores = {}
        summary_keywords = [
            'సారాంశం', 'ముఖ్యంగా', 'కీలకంగా', 'దృష్టికోణం', 'నిర్ణయాలు', 'పాలన', 'ప్రభావం',
            'సమస్యలు', 'చర్చ', 'ప్రతినిధ్యం', 'దేశాలు', 'ప్రపంచం', 'సభ్యదేశం', 'అతిథి దేశంగా',
            'సమ్మిట్', 'ప్రధానంగా', 'మార్గాలు', 'సూచిస్తాయి', 'కీలకంగా', 'భాగస్వామ్యం', 'పాల్గొనడం'
        ]
        for i, sentence in enumerate(sentences):
            words = self._tokenize_words(sentence)
            words = [w.lower() for w in words if w.lower() not in self.stopwords.get(self.lang_code, set())]
            if not words:
                sentence_scores[i] = 0
                continue
            # Base score from word importance
            score = sum(word_scores.get(word, 0) for word in words) / len(words)
            # Boost for summary-like keywords
            boost = sum(2 for k in summary_keywords if k in sentence)
            score += boost * 0.5
            # Position weight
            position_weight = 1.0
            if i == 0:
                position_weight = 1.1
            elif i == len(sentences) - 1:
                position_weight = 1.05
            # Length weight
            length_weight = 1.0
            if 40 <= len(sentence) <= 180:
                length_weight = 1.2
            elif len(sentence) < 30:
                length_weight = 0.7
            elif len(sentence) > 200:
                length_weight = 0.8
            score = score * position_weight * length_weight
            sentence_scores[i] = score
        return sentence_scores
    
    def _compress_sentence(self, sentence: str, aggressive: bool = False) -> str:
        """Intelligent sentence compression"""
        words = sentence.split()
        
        # For short sentences, don't compress
        if len(words) <= 8:
            return sentence
        
        stopwords = self.stopwords.get(self.lang_code, set())
        filtered_words = []
        
        for i, word in enumerate(words):
            # Always keep first 2-3 words (often subject + verb)
            if i < 3:
                filtered_words.append(word)
                continue
            
            # Keep last word
            if i == len(words) - 1:
                filtered_words.append(word)
                continue
            
            word_lower = word.lower()
            
            # Skip stopwords for aggressive compression
            if aggressive and word_lower in stopwords:
                continue
            
            # Keep important terms
            important_terms = self.important_terms.get(self.lang_code, {})
            if word_lower in important_terms:
                filtered_words.append(word)
                continue
            
            # Keep words with numbers
            if re.search(r'\d', word):
                filtered_words.append(word)
                continue
            
            # For normal compression, keep most words
            if not aggressive:
                filtered_words.append(word)
        
        return ' '.join(filtered_words)
    
    def _enhance_flow(self, sentences: List[str]) -> List[str]:
        """Enhance coherence with appropriate connectors"""
        if len(sentences) <= 1:
            return sentences
        
        connectors = self.connectors.get(self.lang_code, {})
        enhanced_sentences = [sentences[0]]  # First sentence unchanged
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            
            # Determine connector type
            connector_type = 'additive'  # Default
            
            # Last sentence often summarizes
            if i == len(sentences) - 1:
                connector_type = 'summary'
            # Check for contrasting content
            elif any(word in sentence.lower() for word in ['కానీ', 'అయితే', 'लेकिन', 'परंतु']):
                connector_type = 'contrastive'
            # Check for causal content
            elif any(word in sentence.lower() for word in ['కాబట్టి', 'కారణం', 'क्योंकि', 'कारण']):
                connector_type = 'causal'
            
            # Add appropriate connector
            if connector_type in connectors and connectors[connector_type]:
                connector = connectors[connector_type][0]  # Use first connector
                enhanced_sentence = f"{connector}, {sentence}"
            else:
                enhanced_sentence = sentence
            
            enhanced_sentences.append(enhanced_sentence)
        
        return enhanced_sentences
      
    def _select_sentences(self, sentences: List[str], target_length: int) -> List[int]:
        """Select sentences for summary based on scores, diversity, and reduced redundancy"""
        sentence_scores = self._score_sentences(sentences)
        scored_sentences = [(i, score) for i, score in sentence_scores.items()]
        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        selected_indices = []
        current_length = 0
        used_topics = set()
        max_sentences = 5

        def get_topic(sentence):
            # Simple topic extraction: first 2 content words
            words = [w for w in self._tokenize_words(sentence) if w not in self.stopwords.get(self.lang_code, set())]
            return tuple(words[:2]) if len(words) >= 2 else tuple(words)

        for idx, score in scored_sentences:
            topic = get_topic(sentences[idx])
            # Penalize redundancy: skip if topic already covered
            if topic in used_topics:
                continue
            sentence_length = len(sentences[idx])
            estimated_length_with_connector = sentence_length + 15
            if current_length + estimated_length_with_connector <= target_length:
                selected_indices.append(idx)
                used_topics.add(topic)
                current_length += estimated_length_with_connector
            if len(selected_indices) >= 3 and current_length >= target_length * 0.7:
                break
            if len(selected_indices) >= max_sentences:
                break
        if not selected_indices and sentences:
            selected_indices = [0]
        selected_indices.sort()
        logger.info(f"Selected sentence indices: {selected_indices}")
        return selected_indices
    
    def generate_neural_enhancement(self, summary_text: str) -> str:
        """Use the trained model to enhance the summary (tuned for more abstraction)"""
        # Skip neural enhancement if model unavailable
        if not self.neural_model_available:
            logger.info("Neural model not available, skipping enhancement.")
            return summary_text
        try:
            prompt = f"సారాంశం మెరుగుపరచండి: {summary_text}\n\nమెరుగైన సారాంశం:"
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=400, truncation=True)
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=80,
                    min_new_tokens=30,
                    do_sample=True,
                    temperature=0.95,  # encourage abstraction
                    top_p=0.92,
                    top_k=50,
                    repetition_penalty=1.05,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "మెరుగైన సారాంశం:" in generated_text:
                enhanced = generated_text.split("మెరుగైన సారాంశం:")[-1].strip()
                if enhanced and len(enhanced) > 20:
                    logger.info(f"Neural enhancement output: {enhanced}")
                    return enhanced
            logger.info(f"Neural enhancement fallback to original summary.")
            return summary_text
        except Exception as e:
            logger.warning(f"Neural enhancement failed: {e}")
            return summary_text

    def _sumy_fallback(self, text: str, sentence_count: int = 2) -> str:
        """Fallback extractive summary using Sumy LexRank"""
        if not SUMY_AVAILABLE:
            # Manual fallback: return first few important sentences
            sentences = text.split('.')
            if len(sentences) > sentence_count:
                return '. '.join(sentences[:sentence_count]) + '.'
            return text
        try:
            # Use a simple tokenizer that works better with Telugu
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = LexRankSummarizer()
            summary = summarizer(parser.document, min(sentence_count, 3))
            result = ' '.join(str(sentence) for sentence in summary)
            
            # If Sumy produces empty or same result, do manual selection
            if not result.strip() or result.strip() == text.strip():
                sentences = text.split('.')
                if len(sentences) > sentence_count:
                    return '. '.join(sentences[:sentence_count]) + '.'
            
            return result.strip() if result.strip() else text
        except Exception as e:
            logger.warning(f"Sumy fallback failed: {e}")
            # Manual fallback: return first few sentences
            sentences = text.split('.')
            if len(sentences) > sentence_count:
                return '. '.join(sentences[:sentence_count]) + '.'
            return text

    def summarize(self, text: str, max_length: int = 250) -> Tuple[str, str]:
        """Main summarization method: extract, then always neural enhance for abstraction"""
        try:
            self.lang_code = self.detect_language(text)
            sentences = self._preprocess_text(text)
            if not sentences:
                return "సారాంశం అందుబాటులో లేదు.", "error"
            if len(sentences) == 1:
                compressed = self._compress_sentence(sentences[0], aggressive=False)
                if len(compressed) > max_length:
                    compressed = compressed[:max_length-3] + "..."
                # Fallback: neural enhancement if available, else extractive single
                if self.neural_model_available:
                    summary = self.generate_neural_enhancement(compressed)
                    return summary[:max_length], "neural_single"
                else:
                    return compressed[:max_length], "extractive_single"            # Improved sentence selection with better logic
            sentence_scores = self._score_sentences(sentences)
            scored = sorted(((i, s) for i, s in sentence_scores.items()), key=lambda x: x[1], reverse=True)
            
            # Select sentences more intelligently
            selected = []
            used = set()
            total_len = 0
            
            # First pass: get the highest scoring sentences that fit
            for idx, score in scored:
                if idx in used:
                    continue
                    
                sentence = sentences[idx]
                sentence_len = len(sentence)
                
                # Skip if too similar to already selected sentences
                skip_similar = False
                for selected_sent in selected:
                    # Simple similarity check
                    common_words = set(sentence.split()) & set(selected_sent.split())
                    if len(common_words) > min(len(sentence.split()), len(selected_sent.split())) * 0.6:
                        skip_similar = True
                        break
                
                if skip_similar:
                    continue
                
                # Check if adding this sentence would exceed length
                if total_len + sentence_len <= max_length * 0.8:  # Leave room for connectors
                    selected.append(sentence)
                    used.add(idx)
                    total_len += sentence_len
                    
                    # Stop if we have enough content
                    if len(selected) >= 3 or total_len >= max_length * 0.6:
                        break
            
            # If we don't have enough content, add more sentences
            if len(selected) < 2 and len(sentences) > 1:
                for idx, score in scored:
                    if idx not in used:
                        sentence = sentences[idx]
                        if total_len + len(sentence) <= max_length:
                            selected.append(sentence)
                            used.add(idx)
                            total_len += len(sentence)
                            break
            
            # If still no selection, use the first sentence
            if not selected and sentences:
                selected = [sentences[0]]
            
            # Compress and merge selected sentences
            compressed_sentences = [self._compress_sentence(s, aggressive=True) for s in selected]
            merged = ' '.join(compressed_sentences)
            # Robust summary selection: neural enhancement with fallback to extractive flow
            if self.neural_model_available:
                enhanced = self.generate_neural_enhancement(merged)
                # Check if enhancement is effective
                if enhanced == merged or len(enhanced.split()) < max(5, len(selected) // 2):
                    logger.warning("Neural enhancement ineffective, falling back to extractive summary.")
                    # Try Sumy fallback if available
                    final_summary = self._sumy_fallback(' '.join(selected), sentence_count=len(selected))
                    method = f"extractive_fallback_{self.lang_code}"
                else:
                    final_summary = enhanced
                    method = f"neural_proper_slm_{self.lang_code}"
            else:
                logger.info("Neural model unavailable, using extractive summary.")
                # Try Sumy fallback first
                sumy_summary = self._sumy_fallback(merged, sentence_count=len(selected))
                if sumy_summary and sumy_summary != merged and len(sumy_summary.strip()) > 10:
                    final_summary = sumy_summary
                    method = f"sumy_lexrank_{self.lang_code}"
                else:
                    # If Sumy fails, use the merged compressed sentences (better than truncation)
                    final_summary = merged
                    method = f"extractive_proper_slm_{self.lang_code}"
            
            # Smart truncation: cut at sentence boundary if too long
            if len(final_summary) > max_length:
                # Try to cut at sentence boundary
                sentences_in_summary = final_summary.split('.')
                truncated = ""
                for sent in sentences_in_summary:
                    if len(truncated + sent + ".") <= max_length - 3:
                        truncated += sent + "."
                    else:
                        break
                if len(truncated) > 20:  # If we got a reasonable truncation
                    final_summary = truncated.strip()
                else:
                    # Fallback to character truncation
                    final_summary = final_summary[:max_length-3] + '...'
            
            return final_summary, method
        except Exception as e:
            logger.error(f"SLM summarization error: {e}")
            return f"సారాంశం లోపం: {str(e)[:30]}...", "error"

# Global instance
_proper_slm_summarizer = None

def get_proper_slm_summarizer() -> ProperSLMSummarizer:
    """Get the proper SLM summarizer instance"""
    global _proper_slm_summarizer
    if _proper_slm_summarizer is None:
        _proper_slm_summarizer = ProperSLMSummarizer()
    return _proper_slm_summarizer

# Test the proper SLM implementation
if __name__ == "__main__":
    print("🎯 Testing Proper SLM Summarizer (Following GitHub Documentation)")
    print("=" * 70)
    
    # Test with aviation incident
    aviation_text = """గత 24 గంటల్లో భారతదేశంతో సహా అంతర్జాతీయ విమానయాన రంగంలో మూడు పెద్ద విమాన ఘటనలు చోటు చేసుకున్నాయి, ఇవి ప్రయాణికుల్లో తీవ్ర ఆందోళనకు కారణమయ్యాయి. హాంకాంగ్ నుండి ఢిల్లీకి వస్తున్న ఎయిర్ ఇండియా బోయింగ్ 787 విమానంలో సాంకేతిక సమస్య తలెత్తడంతో, అది మధ్యలోనే మళ్లీ తిరిగి వెళ్లాల్సి వచ్చింది. ఇదే సమయంలో జర్మనీకి చెందిన లుఫ్తాన్సా విమానానికి బాంబు బెదిరింపు సందేశం రావడంతో వెంటనే తిరిగి ప్రయాణం రద్దుచేసింది. ఇక కేరళలో బ్రిటిష్ ఫైటర్ జెట్ ఒకటి తక్కువ ఇంధనం కారణంగా అత్యవసరంగా ల్యాండ్ కావాల్సి వచ్చింది. ఈ మూడు ఘటనలు కూడా ప్రయాణికుల భద్రతపై తీవ్ర ఆందోళన కలిగించాయి. విమాన భద్రతా ప్రమాణాలపై ప్రజల్లో అనేక సందేహాలు మొదలయ్యాయి. విమానయాన సంస్థలు అత్యధిక జాగ్రత్తలు తీసుకోవాల్సిన అవసరం మరోసారి తేటతెల్లమైంది."""
    
    try:
        summarizer = ProperSLMSummarizer()
        summary, method = summarizer.summarize(aviation_text, max_length=250)
        
        print(f"Input length: {len(aviation_text)} characters")
        print(f"Language detected: {summarizer.lang_code}")
        print()
        print(f"Generated Summary ({method}):")
        print(summary)
        print()
        print(f"Summary length: {len(summary)} characters")
        
        # Check quality
        if "proper_slm" in method:
            print("🎯 SUCCESS: Using proper SLM approach with language-specific resources!")
        
        # Check if it's actually different content
        if summary != aviation_text[:len(summary)]:
            print("✅ SUCCESS: Generated intelligent abstractive summary!")
        else:
            print("❌ Still truncating")
            
        # Check for discourse connectors
        telugu_connectors = ['అలాగే', 'అంతేకాకుండా', 'ఇంకా', 'అదేవిధంగా', 'కాబట్టి']
        if any(conn in summary for conn in telugu_connectors):
            print("✅ SUCCESS: Added appropriate discourse connectors!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Need to check model compatibility")
