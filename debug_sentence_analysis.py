#!/usr/bin/env python3
"""
Debug the sentence scoring and key terms extraction
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

from proper_slm_summarizer import StateOfTheArtIndicSummarizer

def debug_sentence_analysis():
    """Debug sentence scoring and key terms"""
    
    # Initialize summarizer
    summarizer = StateOfTheArtIndicSummarizer()
    
    # User's example text
    text = """ఢిల్లీ నగరం భారతదేశంలో వాయు కాలుష్యానికి కేంద్రంగా మారింది. ముఖ్యంగా శీతాకాలంలో వాయు నాణ్యత తీవ్రంగా పడిపోయి, పౌల్యూషన్‌ కంట్రోల్ బోర్డు సూచించిన ప్రమాణాలను మించి హానికరమైన స్థాయికి చేరుకుంటుంది. పొగమంచు, వాహనాల ఎగురుతున్న పొగ, పరిశ్రమల నుంచి వెలువడే మురికివాయువులు, నిర్మాణ కార్యక్రమాల ధూళి మరియు చుట్టుపక్కల రాష్ట్రాల నుంచి వచ్చే పొలాల కాల్చే దుష్పరిణామాల వలన ఢిల్లీ వాసులు ఆరోగ్య సమస్యలను ఎదుర్కొంటున్నారు. పిల్లలు, వృద్ధులు మరియు శ్వాసకోశ సంబంధిత రుగ్మతలు ఉన్నవారు తీవ్రమైన సమస్యలకు గురవుతున్నారు. ప్రభుత్వం ప్రతీవేళ విపత్కర పరిస్థితుల్లో పాఠశాలలు మూసివేయడం, వాహనాల పరిమిత వినియోగానికి 'ఒడ్ ఈవెన్' విధానం, నిర్మాణాలపై నిషేధం వంటి చర్యలు తీసుకుంటున్నప్పటికీ దీర్ఘకాలిక పరిష్కారం మాత్రం కనిపించడం లేదు. వాతావరణ మార్పులు, జనసాంద్రత మరియు ఆధునిక జీవనశైలిలోని మార్పులు ఈ సమస్యను మరింత వేగంగా పెంచుతున్నాయి. కాలుష్యాన్ని నియంత్రించేందుకు ప్రభుత్వం, పరిశ్రమలు మరియు ప్రజలు సమిష్టిగా వ్యవహరించాల్సిన అవసరం ఎంతో ఉంది."""
    
    sentences = summarizer._preprocess_text(text)
    print(f"Number of sentences: {len(sentences)}")
    
    for i, sentence in enumerate(sentences):
        print(f"\n--- Sentence {i} ---")
        print(f"Text: {sentence}")
        print(f"Length: {len(sentence)} characters")
        
        # Get key terms
        key_terms = summarizer._get_sentence_key_terms(sentence)
        print(f"Key terms: {key_terms}")
        
        # Get sentence score components
        words = summarizer._tokenize_words(sentence)
        stopwords = summarizer.stopwords.get('te', set())
        words_filtered = [w.lower() for w in words if w.lower() not in stopwords]
        print(f"Content words: {words_filtered[:10]}...")  # Show first 10
        
    # Test with larger max_length
    print(f"\n🔍 Testing with larger max_length (400):")
    extractive_summary = summarizer._enhanced_extractive_summarize(sentences, 400)
    print(f"Summary: {extractive_summary}")
    summary_sentences = summarizer._preprocess_text(extractive_summary)
    print(f"Number of sentences in summary: {len(summary_sentences)}")

if __name__ == "__main__":
    debug_sentence_analysis()
