#!/usr/bin/env py
"""
Enhanced Summarization Module for AadiShakthiSLM
Implements better extractive summarization while model trains
"""

import re
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class EnhancedSummarizer:
    """Enhanced extractive summarization for Telugu and Hindi"""
    
    def __init__(self):
        # Important keywords for different languages
        self.important_keywords = {
            "te": [
                # Telugu important terms
                "ప్రభుత్వం", "రాష్ట్రం", "నగరం", "జిల్లా", "ముఖ్యమంత్రి", "ప్రధానమంత్రి",
                "పోలీసులు", "ఆసుపత్రి", "పాఠశాల", "విశ్వవిద్యాలయం", "రైలు", "బస్సు",
                "రోడ్డు", "వర్షం", "వాతావరణం", "వ్యవసాయం", "రైతు", "పంట", "ధాన్యం",
                "బంగారం", "వెండి", "డబ్బు", "రూపాయి", "లక్ష", "కోటి", "వేల", "హజార్",
                "సినిమా", "నటుడు", "నటి", "దర్శకుడు", "టాలీవుడ్", "బాలీవుడ్",
                "వైద్యం", "డాక్టర్", "మందు", "చికిత్స", "ఆరోగ్యం", "వ్యాధి"
            ],
            "hi": [
                # Hindi important terms  
                "सरकार", "राज्य", "शहर", "जिला", "मुख्यमंत्री", "प्रधानमंत्री",
                "पुलिस", "अस्पताल", "स्कूल", "विश्वविद्यालय", "ट्रेन", "बस",
                "सड़क", "बारिश", "मौसम", "कृषि", "किसान", "फसल", "अनाज",
                "सोना", "चांदी", "पैसा", "रुपया", "लाख", "करोड़", "हजार",
                "फिल्म", "अभिनेता", "अभिनेत्री", "निर्देशक", "बॉलीवुड",
                "चिकित्सा", "डॉक्टर", "दवा", "इलाज", "स्वास्थ्य", "बीमारी"
            ],
            "en": [
                "government", "state", "city", "district", "minister", "prime",
                "police", "hospital", "school", "university", "train", "bus",
                "road", "rain", "weather", "agriculture", "farmer", "crop",
                "gold", "silver", "money", "rupee", "lakh", "crore", "thousand",
                "movie", "actor", "actress", "director", "bollywood",
                "medical", "doctor", "medicine", "treatment", "health", "disease"
            ]
        }
    
    def detect_language(self, text: str) -> str:
        """Simple language detection"""
        # Telugu Unicode range
        telugu_chars = sum(1 for c in text if 0x0C00 <= ord(c) <= 0x0C7F)
        # Hindi/Devanagari Unicode range  
        hindi_chars = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)
        # English characters
        english_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        
        total_chars = telugu_chars + hindi_chars + english_chars
        if total_chars == 0:
            return "en"
        
        if telugu_chars / total_chars > 0.3:
            return "te"
        elif hindi_chars / total_chars > 0.3:
            return "hi"
        else:
            return "en"
    
    def split_sentences(self, text: str, language: str) -> List[str]:
        """Split text into sentences based on language"""
        if language in ["te", "hi"]:
            # Split on Devanagari punctuation and common punctuation
            sentences = re.split(r'[।॥.!?]+', text)
        else:
            # English sentence splitting
            sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter sentences
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        return sentences
    
    def calculate_sentence_score(self, sentence: str, word_freq: Dict[str, int], language: str) -> float:
        """Calculate importance score for a sentence"""
        words = sentence.lower().split()
        if not words:
            return 0.0
        
        # Base score from word frequency
        freq_score = sum(word_freq.get(word, 0) for word in words) / len(words)
        
        # Bonus for important keywords
        keyword_bonus = 0
        important_words = self.important_keywords.get(language, [])
        for word in words:
            if word in important_words:
                keyword_bonus += 1
        
        # Normalize keyword bonus
        keyword_score = keyword_bonus / len(words) if words else 0
        
        # Position bonus (first and last sentences are often important)
        position_score = 0
        
        # Length penalty (very short or very long sentences are less preferred)
        length_penalty = 0
        if len(words) < 5:
            length_penalty = -0.2
        elif len(words) > 30:
            length_penalty = -0.1
        
        # Numerical content bonus (numbers often indicate important facts)
        number_bonus = 0
        for word in words:
            if any(c.isdigit() for c in word):
                number_bonus += 0.1
        
        # Final score
        final_score = (freq_score * 0.4 + 
                      keyword_score * 0.3 + 
                      position_score * 0.1 + 
                      number_bonus * 0.2 + 
                      length_penalty)
        
        return final_score
    
    def extract_key_information(self, text: str, language: str) -> Dict[str, List[str]]:
        """Extract key information categories"""
        info = {
            "numbers": [],
            "locations": [],
            "names": [],
            "events": []
        }
        
        # Extract numbers and quantities
        number_patterns = [
            r'\d+\.?\d*\s*(?:లక్ష|కోటి|వేల|हजार|लाख|करोड़|thousand|lakh|crore)',
            r'\d+\.?\d*\s*(?:రూపాయ|रुपय|rupee)',
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',  # dates
            r'\d+\.?\d*\s*(?:కిలో|किलो|kg|టన|टन|ton)'
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            info["numbers"].extend(matches)
        
        # Extract locations (basic patterns)
        if language == "te":
            location_patterns = [
                r'[ա-ౌ]+(?:పురం|నగర్|బాద్|గ్రామం)',
                r'(?:హైదరాబాద్|వరంగల్|విజయవాడ|తిరుపతి|గుంటూర్)'
            ]
        elif language == "hi":
            location_patterns = [
                r'[अ-ह]+(?:पुर|नगर|बाद|गांव)',
                r'(?:दिल्ली|मुंबई|कोलकाता|चेन्नई|बेंगलुरू)'
            ]
        else:
            location_patterns = [
                r'[A-Z][a-z]+(?:pur|bad|nagar|city)',
                r'(?:Delhi|Mumbai|Kolkata|Chennai|Bangalore|Hyderabad)'
            ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            info["locations"].extend(matches)
        
        return info
    
    def summarize(self, text: str, max_sentences: int = 3) -> str:
        """Enhanced extractive summarization"""
        if not text or len(text.strip()) < 50:
            return text.strip()
        
        # Detect language
        language = self.detect_language(text)
        
        # Split into sentences
        sentences = self.split_sentences(text, language)
        
        if len(sentences) <= max_sentences:
            return " ".join(sentences)
        
        # Calculate word frequency
        words = text.lower().split()
        word_freq = Counter(words)
        
        # Remove very common words
        common_words = {"और", "या", "के", "की", "को", "में", "से", "पर", "for", "the", "and", "or", "in", "on", "at", "to", "of"}
        for word in common_words:
            if word in word_freq:
                del word_freq[word]
        
        # Calculate sentence scores
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = self.calculate_sentence_score(sentence, word_freq, language)
            
            # Position bonus
            if i == 0:  # First sentence
                score += 0.2
            elif i == len(sentences) - 1:  # Last sentence
                score += 0.1
            
            sentence_scores.append((score, i, sentence))
        
        # Sort by score and select top sentences
        sentence_scores.sort(reverse=True, key=lambda x: x[0])
        selected_sentences = sentence_scores[:max_sentences]
        
        # Sort selected sentences by original order
        selected_sentences.sort(key=lambda x: x[1])
        
        # Create summary
        summary_sentences = [sentence for _, _, sentence in selected_sentences]
        summary = " ".join(summary_sentences)
        
        # Extract key information for enhancement
        key_info = self.extract_key_information(text, language)
        
        logger.info(f"Enhanced summarization - Language: {language}, "
                   f"Original sentences: {len(sentences)}, "
                   f"Selected: {max_sentences}, "
                   f"Key numbers: {len(key_info['numbers'])}")
        
        return summary

# Global instance
enhanced_summarizer = EnhancedSummarizer()

def enhance_summarization(text: str, max_sentences: int = 3) -> Dict:
    """Enhanced summarization function for API use"""
    
    # Detect language first
    language = enhanced_summarizer.detect_language(text)
    
    # Generate enhanced summary
    summary = enhanced_summarizer.summarize(text, max_sentences)
    
    # Extract additional information
    key_info = enhanced_summarizer.extract_key_information(text, language)
    
    return {
        "summary": summary,
        "language": language,
        "method": "enhanced_extractive",
        "key_information": key_info,
        "original_length": len(text.split()),
        "summary_length": len(summary.split()),
        "compression_ratio": len(summary.split()) / len(text.split()) if text.split() else 0
    }
