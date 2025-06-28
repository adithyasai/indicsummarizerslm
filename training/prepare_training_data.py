#!/usr/bin/env python3
"""
Data Preparation for Incremental Training
Processes and organizes Telugu and Hindi datasets for efficient training
"""

import os
import json
import gzip
import logging
from pathlib import Path
from typing import List, Dict, Iterator
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    """Process and prepare training data"""
    
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Text cleaning patterns
        self.clean_patterns = [
            (r'\s+', ' '),                    # Multiple spaces to single space
            (r'\n+', ' '),                    # Multiple newlines to space
            (r'[^\u0900-\u097F\u0C00-\u0C7F\u0020-\u007E]+', ''),  # Keep only Devanagari, Telugu, and basic ASCII
            (r'^\s+|\s+$', ''),               # Strip leading/trailing spaces
        ]
        
        # Minimum and maximum text lengths
        self.min_length = 50
        self.max_length = 1000
        
        logger.info(f"✅ Data processor initialized")
        logger.info(f"   Output directory: {output_dir}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Apply cleaning patterns
        for pattern, replacement in self.clean_patterns:
            text = re.sub(pattern, replacement, text)
        
        return text.strip()
    
    def is_valid_text(self, text: str) -> bool:
        """Check if text is valid for training"""
        if not text or len(text) < self.min_length or len(text) > self.max_length:
            return False
        
        # Check if text contains Telugu or Hindi characters
        telugu_chars = len(re.findall(r'[\u0C00-\u0C7F]', text))
        hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
        
        # At least 10% should be Indian language characters
        total_chars = len(text)
        indian_char_ratio = (telugu_chars + hindi_chars) / total_chars
        
        return indian_char_ratio >= 0.1
    
    def process_wikipedia_data(self, data_dir: str, language: str) -> List[str]:
        """Process Wikipedia data"""
        logger.info(f"📖 Processing Wikipedia data for {language}")
        
        texts = []
        wiki_dir = os.path.join(data_dir, f"{language}_wikipedia")
        
        if not os.path.exists(wiki_dir):
            logger.warning(f"Wikipedia directory not found: {wiki_dir}")
            return texts
        
        # Look for text files
        for file_path in Path(wiki_dir).rglob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Split into paragraphs
                    paragraphs = content.split('\n\n')
                    
                    for para in paragraphs:
                        cleaned = self.clean_text(para)
                        if self.is_valid_text(cleaned):
                            texts.append(cleaned)
                            
                        # Limit per file to avoid memory issues
                        if len(texts) >= 500:
                            break
                            
                if len(texts) >= 2000:  # Limit total per language
                    break
                    
            except Exception as e:
                logger.warning(f"Could not process {file_path}: {e}")
        
        logger.info(f"   Processed {len(texts)} Wikipedia texts for {language}")
        return texts
    
    def process_cc100_data(self, data_dir: str, language: str) -> List[str]:
        """Process CC-100 data"""
        logger.info(f"🌐 Processing CC-100 data for {language}")
        
        texts = []
        cc100_dir = os.path.join(data_dir, f"cc100_{language}")
        
        if not os.path.exists(cc100_dir):
            logger.warning(f"CC-100 directory not found: {cc100_dir}")
            return texts
        
        # Look for compressed or text files
        for file_path in Path(cc100_dir).rglob("*"):
            if file_path.suffix in ['.txt', '.gz']:
                try:
                    if file_path.suffix == '.gz':
                        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                            content = f.read()
                    else:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                    
                    # Split into lines and process
                    lines = content.split('\n')
                    
                    for line in lines:
                        cleaned = self.clean_text(line)
                        if self.is_valid_text(cleaned):
                            texts.append(cleaned)
                            
                        # Limit per file
                        if len(texts) >= 1000:
                            break
                            
                    if len(texts) >= 3000:  # Limit total
                        break
                        
                except Exception as e:
                    logger.warning(f"Could not process {file_path}: {e}")
        
        logger.info(f"   Processed {len(texts)} CC-100 texts for {language}")
        return texts
    
    def process_xlsum_data(self, data_dir: str) -> List[str]:
        """Process XL-Sum data (summaries)"""
        logger.info(f"📰 Processing XL-Sum data")
        
        texts = []
        
        # Look for XL-Sum files
        xlsum_files = [
            os.path.join(data_dir, "xlsum_telugu_extended.json"),
            os.path.join(data_dir, "xlsum_hindi_extended.json"),
            os.path.join(data_dir, "xlsum_data.json"),
        ]
        
        for file_path in xlsum_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    if isinstance(data, list):
                        for item in data:
                            # Extract text and summary
                            if isinstance(item, dict):
                                text = item.get('text', '') or item.get('document', '')
                                summary = item.get('summary', '')
                                
                                # Process both text and summary
                                for content in [text, summary]:
                                    cleaned = self.clean_text(content)
                                    if self.is_valid_text(cleaned):
                                        texts.append(cleaned)
                                        
                                if len(texts) >= 2000:
                                    break
                                    
                except Exception as e:
                    logger.warning(f"Could not process {file_path}: {e}")
        
        logger.info(f"   Processed {len(texts)} XL-Sum texts")
        return texts
    
    def process_samanantar_data(self, data_dir: str) -> List[str]:
        """Process Samanantar parallel data"""
        logger.info(f"🔄 Processing Samanantar data")
        
        texts = []
        
        # Look for Samanantar files
        samanantar_files = [
            os.path.join(data_dir, "samanantar_extended.json"),
            os.path.join(data_dir, "samanantar_data.json"),
        ]
        
        for file_path in samanantar_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                # Extract source and target texts
                                source = item.get('source', '') or item.get('en', '')
                                target = item.get('target', '') or item.get('hi', '') or item.get('te', '')
                                
                                # We mainly want the Indian language text
                                for content in [target, source]:
                                    cleaned = self.clean_text(content)
                                    if self.is_valid_text(cleaned):
                                        texts.append(cleaned)
                                        
                                if len(texts) >= 2000:
                                    break
                                    
                except Exception as e:
                    logger.warning(f"Could not process {file_path}: {e}")
        
        logger.info(f"   Processed {len(texts)} Samanantar texts")
        return texts
    
    def create_mixed_training_data(self, telugu_texts: List[str], hindi_texts: List[str]) -> List[str]:
        """Create mixed training data with both languages"""
        logger.info("🎯 Creating mixed training data")
        
        mixed_texts = []
        
        # Interleave Telugu and Hindi texts
        max_len = max(len(telugu_texts), len(hindi_texts))
        
        for i in range(max_len):
            if i < len(telugu_texts):
                mixed_texts.append(telugu_texts[i])
            if i < len(hindi_texts):
                mixed_texts.append(hindi_texts[i])
        
        # Shuffle to mix better
        import random
        random.shuffle(mixed_texts)
        
        logger.info(f"   Created {len(mixed_texts)} mixed texts")
        return mixed_texts
    
    def save_processed_data(self, texts: List[str], filename: str):
        """Save processed data to JSON file"""
        output_path = os.path.join(self.output_dir, filename)
        
        # Save with metadata
        data = {
            'texts': texts,
            'count': len(texts),
            'total_chars': sum(len(text) for text in texts),
            'avg_length': sum(len(text) for text in texts) / len(texts) if texts else 0
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Saved {len(texts)} texts to {output_path}")
        
        # Also save as simple text list for compatibility
        simple_path = output_path.replace('.json', '_simple.json')
        with open(simple_path, 'w', encoding='utf-8') as f:
            json.dump(texts, f, ensure_ascii=False, indent=2)
    
    def process_all_data(self, raw_data_dir: str = "data/raw"):
        """Process all available data"""
        logger.info("🔄 Processing all training data...")
        logger.info("=" * 50)
        
        # Process Telugu data
        logger.info("\n📝 Processing Telugu data...")
        telugu_texts = []
        telugu_texts.extend(self.process_wikipedia_data(raw_data_dir, "telugu"))
        telugu_texts.extend(self.process_cc100_data(raw_data_dir, "telugu"))
        
        # Process Hindi data
        logger.info("\n📝 Processing Hindi data...")
        hindi_texts = []
        hindi_texts.extend(self.process_wikipedia_data(raw_data_dir, "hindi"))
        hindi_texts.extend(self.process_cc100_data(raw_data_dir, "hindi"))
        
        # Process multilingual data
        logger.info("\n📝 Processing multilingual data...")
        xlsum_texts = self.process_xlsum_data(raw_data_dir)
        samanantar_texts = self.process_samanantar_data(raw_data_dir)
        
        # Combine with main datasets
        telugu_texts.extend([t for t in xlsum_texts if self.is_telugu_text(t)])
        hindi_texts.extend([t for t in xlsum_texts if self.is_hindi_text(t)])
        
        telugu_texts.extend([t for t in samanantar_texts if self.is_telugu_text(t)])
        hindi_texts.extend([t for t in samanantar_texts if self.is_hindi_text(t)])
        
        # Remove duplicates
        telugu_texts = list(set(telugu_texts))
        hindi_texts = list(set(hindi_texts))
        
        # Create mixed training data
        mixed_texts = self.create_mixed_training_data(telugu_texts, hindi_texts)
        
        # Save all datasets
        logger.info("\n💾 Saving processed datasets...")
        self.save_processed_data(telugu_texts, "telugu_texts.json")
        self.save_processed_data(hindi_texts, "hindi_texts.json")
        self.save_processed_data(mixed_texts, "mixed_texts.json")
        
        # Summary
        logger.info("\n📊 Data Processing Summary:")
        logger.info(f"   Telugu texts: {len(telugu_texts):,}")
        logger.info(f"   Hindi texts: {len(hindi_texts):,}")
        logger.info(f"   Mixed texts: {len(mixed_texts):,}")
        logger.info(f"   Total training examples: {len(telugu_texts) + len(hindi_texts) + len(mixed_texts):,}")
        
        return {
            'telugu': len(telugu_texts),
            'hindi': len(hindi_texts),
            'mixed': len(mixed_texts)
        }
    
    def is_telugu_text(self, text: str) -> bool:
        """Check if text is primarily Telugu"""
        telugu_chars = len(re.findall(r'[\u0C00-\u0C7F]', text))
        total_chars = len(text)
        return telugu_chars / total_chars > 0.3 if total_chars > 0 else False
    
    def is_hindi_text(self, text: str) -> bool:
        """Check if text is primarily Hindi"""
        hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
        total_chars = len(text)
        return hindi_chars / total_chars > 0.3 if total_chars > 0 else False

def create_sample_data_if_missing():
    """Create sample data if no raw data is available"""
    logger.info("🎯 Creating sample training data...")
    
    sample_texts = {
        'telugu': [
            "తెలుగు భాష భారతదేశంలో అత్యంత ప్రాచీనమైన భాషలలో ఒకటి. ఇది దక్షిణ భారతదేశంలో మాట్లాడబడుతుంది.",
            "తెలంగాణ రాష్ట్రం 2014లో ఏర్పడింది. హైదరాబాద్ దాని రాజధాని.",
            "తెలుగు సాహిత్యం చాలా గొప్పది. అనేక మంది కవులు మరియు రచయితలు ఉన్నారు.",
            "వాతావరణ మార్పులు ప్రపంచవ్యాప్తంగా ప్రభావం చూపిస్తున్నాయి. భారతదేశంలో కూడా ఈ మార్పులు కనిపిస్తున్నాయి.",
            "ప్రయుక్తిశాస్త్రం వేగంగా అభివృద్ధి చెందుతోంది. కృత్రిమ మేధస్సు మనం చేసే పనులను మార్చుతోంది.",
        ],
        'hindi': [
            "हिंदी भारत की राजभाषा है और यह देवनागरी लिपि में लिखी जाती है।",
            "भारत एक विविधताओं से भरा देश है जहाँ अनेक भाषाएँ और संस्कृतियाँ पनपती हैं।",
            "तकनीक के क्षेत्र में भारत तेजी से आगे बढ़ रहा है। डिजिटल इंडिया अभियान इसका प्रमाण है।",
            "शिक्षा हर व्यक्ति का मौलिक अधिकार है। गुणवत्तापूर्ण शिक्षा समाज का आधार है।",
            "पर्यावरण संरक्षण आज की सबसे बड़ी चुनौती है। हमें प्रकृति का सम्मान करना चाहिए।",
        ]
    }
    
    # Create more samples by variation
    extended_samples = {
        'telugu': [],
        'hindi': []
    }
    
    # Extend Telugu samples
    for base_text in sample_texts['telugu']:
        extended_samples['telugu'].append(base_text)
        # Add variations
        extended_samples['telugu'].append(f"నేటి రోజు {base_text}")
        extended_samples['telugu'].append(f"{base_text} ఇది చాలా ముఖ్యమైన విషయం.")
    
    # Extend Hindi samples
    for base_text in sample_texts['hindi']:
        extended_samples['hindi'].append(base_text)
        # Add variations
        extended_samples['hindi'].append(f"आज के दिन {base_text}")
        extended_samples['hindi'].append(f"{base_text} यह बहुत महत्वपूर्ण बात है।")
    
    # Multiply to create more training data
    final_samples = {
        'telugu': extended_samples['telugu'] * 50,  # 750 samples
        'hindi': extended_samples['hindi'] * 50    # 750 samples
    }
    
    return final_samples

def main():
    """Main data preparation function"""
    logger.info("🚀 AadiShakti SLM - Data Preparation")
    logger.info("=" * 60)
    
    # Create data processor
    processor = DataProcessor()
    
    # Check if raw data exists
    raw_data_dir = "data/raw"
    if not os.path.exists(raw_data_dir) or not any(os.listdir(raw_data_dir)):
        logger.info("📝 No raw data found, creating sample data...")
        sample_data = create_sample_data_if_missing()
        
        # Save sample data
        processor.save_processed_data(sample_data['telugu'], "telugu_texts.json")
        processor.save_processed_data(sample_data['hindi'], "hindi_texts.json")
        
        # Create mixed data
        mixed = processor.create_mixed_training_data(sample_data['telugu'], sample_data['hindi'])
        processor.save_processed_data(mixed, "mixed_texts.json")
        
        logger.info("✅ Sample data created and saved")
    else:
        # Process existing raw data
        logger.info("📚 Processing existing raw data...")
        result = processor.process_all_data(raw_data_dir)
        logger.info("✅ Data processing completed")
    
    logger.info("\n🎉 Data preparation completed!")
    logger.info("   Ready for incremental training")

if __name__ == "__main__":
    main()
