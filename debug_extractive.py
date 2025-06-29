#!/usr/bin/env py    # Sample Telugu text (user's example)
    text = """ఢిల్లీ నగరం భారతదేశంలో వాయు కాలుష్యానికి కేంద్రంగా మారింది. ముఖ్యంగా శీతాకాలంలో వాయు నాణ్యత తీవ్రంగా పడిపోయి, పౌల్యూషన్‌ కంట్రోల్ బోర్డు సూచించిన ప్రమాణాలను మించి హానికరమైన స్థాయికి చేరుకుంటుంది. పొగమంచు, వాహనాల ఎగురుతున్న పొగ, పరిశ్రమల నుంచి వెలువడే మురికివాయువులు, నిర్మాణ కార్యక్రమాల ధూళి మరియు చుట్టుపక్కల రాష్ట్రాల నుంచి వచ్చే పొలాల కాల్చే దుష్పరిణామాల వలన ఢిల్లీ వాసులు ఆరోగ్య సమస్యలను ఎదుర్కొంటున్నారు. పిల్లలు, వృద్ధులు మరియు శ్వాసకోశ సంబంధిత రుగ్మతలు ఉన్నవారు తీవ్రమైన సమస్యలకు గురవుతున్నారు. ప్రభుత్వం ప్రతీవేళ విపత్కర పరిస్థితుల్లో పాఠశాలలు మూసివేయడం, వాహనాల పరిమిత వినియోగానికి 'ఒడ్ ఈవెన్' విధానం, నిర్మాణాలపై నిషేధం వంటి చర్యలు తీసుకుంటున్నప్పటికీ దీర్ఘకాలిక పరిష్కారం మాత్రం కనిపించడం లేదు. వాతావరణ మార్పులు, జనసాంద్రత మరియు ఆధునిక జీవనశైలిలోని మార్పులు ఈ సమస్యను మరింత వేగంగా పెంచుతున్నాయి. కాలుష్యాన్ని నియంత్రించేందుకు ప్రభుత్వం, పరిశ్రమలు మరియు ప్రజలు సమిష్టిగా వ్యవహరించాల్సిన అవసరం ఎంతో ఉంది."""""
Debug script to test extractive summarization
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

from proper_slm_summarizer import StateOfTheArtIndicSummarizer

def test_extractive_selection():
    """Test extractive sentence selection"""
    
    # Initialize summarizer
    summarizer = StateOfTheArtIndicSummarizer()
    
    # Sample Telugu text
    text = """
    జి20 సమ్మిట్‌కు అతిథి దేశంగా భారత్ ముఖ్యమైన పాత్ర పోషిస్తోంది. ఈ సమ్మిట్‌లో 19 దేశాలు మరియు యూరోపియన్ యూనియన్ పాల్గొంటాయి. 
    ప్రపంచ ఆర్థిక వ్యవస్థ, వాతావరణ మార్పులు, మరియు డిజిటల్ రూపాంతరం వంటి అంశాలపై చర్చలు జరుగుతాయి.
    భారత్ అధ్యక్షతన జరిగే ఈ సమ్మిట్ అనేక కీలక నిర్ణయాలను తీసుకోవాలని అంచనా వేస్తున్నారు.
    దేశాల మధ్య సహకారం పెంచడం మరియు శాంతి స్థాపనకు ఈ సమ్మిట్ దోహదపడుతుందని నిపுణులు అభిప్రాయపడుతున్నారు.
    వాణిజ్య రంగంలో కొత్త అవకాశాలు కల్పించడం కూడా ముఖ్య లక్ష్యం.
    """.strip()
    
    # Split into sentences
    sentences = summarizer._preprocess_text(text)
    print(f"Number of sentences: {len(sentences)}")
    print("\nSentences:")
    for i, sent in enumerate(sentences):
        print(f"{i}: {sent}")
    
    # Score sentences
    scores = summarizer._score_sentences(sentences)
    print(f"\nSentence scores:")
    for i, score in scores.items():
        print(f"{i}: {score:.3f} - {sentences[i][:50]}...")
    
    # Test extractive summarization
    summary = summarizer._enhanced_extractive_summarize(sentences, 300)
    print(f"\nExtractive summary:")
    print(summary)
    print(f"\nSummary length: {len(summary)}")
    
    # Count sentences in summary
    summary_sentences = summarizer._preprocess_text(summary)
    print(f"Number of sentences in summary: {len(summary_sentences)}")

if __name__ == "__main__":
    test_extractive_selection()
