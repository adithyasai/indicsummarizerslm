#!/usr/bin/env python3
"""
Debug script to test extractive summarization with user's example
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

from proper_slm_summarizer import StateOfTheArtIndicSummarizer

def test_user_example():
    """Test with user's specific example"""
    
    # Initialize summarizer
    summarizer = StateOfTheArtIndicSummarizer()
    
    # User's example text
    text = """ఢిల్లీ నగరం భారతదేశంలో వాయు కాలుష్యానికి కేంద్రంగా మారింది. ముఖ్యంగా శీతాకాలంలో వాయు నాణ్యత తీవ్రంగా పడిపోయి, పౌల్యూషన్‌ కంట్రోల్ బోర్డు సూచించిన ప్రమాణాలను మించి హానికరమైన స్థాయికి చేరుకుంటుంది. పొగమంచు, వాహనాల ఎగురుతున్న పొగ, పరిశ్రమల నుంచి వెలువడే మురికివాయువులు, నిర్మాణ కార్యక్రమాల ధూళి మరియు చుట్టుపక్కల రాష్ట్రాల నుంచి వచ్చే పొలాల కాల్చే దుష్పరిణామాల వలన ఢిల్లీ వాసులు ఆరోగ్య సమస్యలను ఎదుర్కొంటున్నారు. పిల్లలు, వృద్ధులు మరియు శ్వాసకోశ సంబంధిత రుగ్మతలు ఉన్నవారు తీవ్రమైన సమస్యలకు గురవుతున్నారు. ప్రభుత్వం ప్రతీవేళ విపత్కర పరిస్థితుల్లో పాఠశాలలు మూసివేయడం, వాహనాల పరిమిత వినియోగానికి 'ఒడ్ ఈవెన్' విధానం, నిర్మాణాలపై నిషేధం వంటి చర్యలు తీసుకుంటున్నప్పటికీ దీర్ఘకాలిక పరిష్కారం మాత్రం కనిపించడం లేదు. వాతావరణ మార్పులు, జనసాంద్రత మరియు ఆధునిక జీవనశైలిలోని మార్పులు ఈ సమస్యను మరింత వేగంగా పెంచుతున్నాయి. కాలుష్యాన్ని నియంత్రించేందుకు ప్రభుత్వం, పరిశ్రమలు మరియు ప్రజలు సమిష్టిగా వ్యవహరించాల్సిన అవసరం ఎంతో ఉంది."""
    
    print("🔍 Testing with user's example text")
    print(f"Original text length: {len(text)} characters")
    
    # Test full summarization
    print("\n📝 Testing full summarization...")
    try:
        summary, method = summarizer.summarize(text, max_length=150)
        print(f"✅ Summary (method: {method}):")
        print(f"   {summary}")
        print(f"   Length: {len(summary)} characters")
    except Exception as e:
        print(f"❌ Full summarization failed: {e}")
    
    # Test extractive only
    print("\n🎯 Testing extractive summarization only...")
    sentences = summarizer._preprocess_text(text)
    print(f"Number of sentences: {len(sentences)}")
    
    scores = summarizer._score_sentences(sentences)
    print(f"\nTop 3 sentence scores:")
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for i, (idx, score) in enumerate(sorted_scores[:3]):
        print(f"{i+1}. Score {score:.3f}: {sentences[idx][:80]}...")
    
    extractive_summary = summarizer._enhanced_extractive_summarize(sentences, 150)
    print(f"\n✅ Extractive summary:")
    print(f"   {extractive_summary}")
    print(f"   Length: {len(extractive_summary)} characters")
    
    # Count sentences in extractive summary
    summary_sentences = summarizer._preprocess_text(extractive_summary)
    print(f"   Number of sentences: {len(summary_sentences)}")

if __name__ == "__main__":
    test_user_example()
