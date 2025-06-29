#!/usr/bin/env python3
"""
Debug the spacing/formatting issues in the summary output
"""

from proper_slm_summarizer import StateOfTheArtIndicSummarizer

def debug_summary_formatting():
    """Debug the actual summary content vs display issues"""
    
    text = """భారతదేశం అనేది ప్రపంచంలో అత్యంత వైవిధ్యభరిత దేశాలలో ఒకటిగా గుర్తించబడింది. భాష, సంస్కృతి, మతం, వేషభాష, ఆచారాలు, భోజన అలవాట్లు మొదలైన అనేక అంశాలలో భారతదేశంలో విభిన్నత స్పష్టంగా కనిపిస్తుంది. ప్రతి రాష్ట్రానికి తాను ప్రత్యేకంగా అభివృద్ధిచేసుకున్న సంప్రదాయాలు, పండుగలు, భాషలు ఉంటాయి. ఉదాహరణకు, ఉత్తర భారతదేశంలో హిందీ ఎక్కువగా మాట్లాడితే, దక్షిణ భారతదేశంలో తెలుగు, తమిళం, కన్నడ, మలయాళం వంటి భాషలు మాట్లాడతారు. మతపరంగా కూడా హిందూమతం, ఇస్లాం, క్రైస్తవం, సిక్కిజం, బౌద్ధం, జైనిజం లాంటి మతాలు శాంతియుతంగా సహజీవనం చేస్తూ దేశ సంస్కృతిని మరింత పరిపుష్టిగా మార్చుతున్నాయి. ఈ విభిన్నత మధ్యలో ఏకత్వ భావనను కలిగించడం భారతదేశ ప్రత్యేకత. 'ఏకతలో అనేకత' అనే సూత్రాన్ని ఆచరిస్తూ, భారత ప్రజలు తమ విభిన్నతను గౌరవించుకుంటూ ఒక దేశంగా ముందుకెళ్తున్నారు. భారత వైవిధ్యమంతా దేశ సంస్కృతిని, చరిత్రను, ఐక్యతను ప్రతిబింబించే శక్తివంతమైన రూపకల్పనగా నిలిచింది."""
    
    print("🔍 DEBUGGING SUMMARY FORMATTING ISSUES")
    print("=" * 60)
    
    summarizer = StateOfTheArtIndicSummarizer()
    summary, method = summarizer.summarize(text, max_length=350)
    
    print(f"📝 Raw summary length: {len(summary)}")
    print(f"🔧 Method: {method}")
    print()
    
    # Show raw bytes to check for hidden characters
    print("🔍 Raw summary bytes:")
    print(repr(summary))
    print()
    
    # Show character by character analysis
    print("🔍 Character analysis:")
    for i, char in enumerate(summary):
        if ord(char) < 32 or ord(char) > 126 and ord(char) < 0x0C00:  # Control chars or non-Telugu
            print(f"   Position {i}: '{char}' (ord: {ord(char)})")
    
    # Show clean version
    print()
    print("🧹 Clean version (removing extra spaces):")
    clean_summary = ' '.join(summary.split())
    print(clean_summary)
    print()
    print(f"📏 Clean length: {len(clean_summary)}")
    
    # Check individual sentences
    print()
    print("📋 Sentence breakdown:")
    sentences = summarizer._preprocess_text(summary)
    for i, sentence in enumerate(sentences, 1):
        print(f"   {i}. '{sentence.strip()}'")
        print(f"      Length: {len(sentence.strip())}")

if __name__ == "__main__":
    debug_summary_formatting()
