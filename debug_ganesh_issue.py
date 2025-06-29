#!/usr/bin/env python3
"""
Quick debugging script for the Ganesh Chaturthi text issue
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

from proper_slm_summarizer import StateOfTheArtIndicSummarizer

def debug_ganesh_text():
    """Debug the specific Ganesh Chaturthi text issue"""
    
    print("🐛 DEBUGGING GANESH CHATURTHI TEXT ISSUE")
    print("=" * 60)
    
    # Your problematic text
    text = """గణేశ్ చతుర్థి అనేది హిందూ మతంలో అత్యంత ప్రాముఖ్యత కలిగిన పండుగలలో ఒకటి. ఈ పండుగను భగవంతుడు వినాయకుడి జన్మదినంగా ప్రతి సంవత్సరం భాద్రపద శుద్ధ చతుర్థి నాడు ఘనంగా జరుపుకుంటారు. గణపతి దేవుడు విజ్ఞానానికి, విజయానికి, ఆటంకాల నివారణకు ప్రతీకగా పూజించబడతాడు. ఈ పండుగ రోజున ఇంటిల్లిపాదీ గణపతి విగ్రహాన్ని స్థాపించి, పుష్పాలు, పళ్ళు, మోదకాలు (కుదురు లడ్డూలు) వంటి నైవేద్యాలతో గణేశుడిని భక్తితో పూజిస్తారు. దేశవ్యాప్తంగా, ముఖ్యంగా మహారాష్ట్ర, ఆంధ్రప్రదేశ్, తెలంగాణ, తమిళనాడు, కర్ణాటక వంటి రాష్ట్రాలలో ఈ పండుగను అత్యంత ఉత్సాహంగా జరుపుకుంటారు. పెద్ద పెద్ద పండుగ మండపాలలో భారీ విగ్రహాలను ప్రతిష్టించి, పలు రోజుల పాటు నృత్యాలు, సంగీత కార్యక్రమాలు, పూజలు నిర్వహిస్తారు. చివరగా, వినాయక నిమజ్జనంతో ఈ పండుగ ముగుస్తుంది. ఇది సామూహిక భక్తి, సాంస్కృతిక ఐక్యతను ప్రతిబింబించే పండుగగా మారింది. గణేశ్ చతుర్థి పండుగ సమయంలో పర్యావరణాన్ని దృష్టిలో ఉంచుకుని మట్టి విగ్రహాలను ప్రోత్సహించడం, పచ్చదనాన్ని కాపాడటం గురించి కూడా ప్రజల్లో చైతన్యం పెరుగుతోంది."""
    
    print(f"📋 Original text: {len(text)} characters")
    
    # Initialize summarizer
    summarizer = StateOfTheArtIndicSummarizer()
    
    # Test with different max_lengths
    for max_len in [200, 250, 300]:
        print(f"\n📏 Testing with max_length = {max_len}")
        
        try:
            summary, method = summarizer.summarize(text, max_length=max_len)
            
            print(f"   🔧 Method: {method}")
            print(f"   📊 Length: {len(summary)} characters")
            print(f"   📝 Summary:")
            print(f"      {summary}")
            
            # Check if it's the fragmentation issue
            if "..." in summary:
                print(f"   ⚠️  Contains ellipsis - truncation happening")
            
            # Check if it's copying fragments
            original_sentences = text.split('.')
            is_copying = False
            for orig_sent in original_sentences:
                if len(orig_sent.strip()) > 30 and orig_sent.strip() in summary:
                    print(f"   ❌ FOUND COPYING: '{orig_sent.strip()[:50]}...'")
                    is_copying = True
                    break
            
            if not is_copying:
                print(f"   ✅ No direct copying detected")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n🔍 Let's examine the sentence processing...")
    sentences = summarizer._preprocess_text(text)
    print(f"📝 Text split into {len(sentences)} sentences:")
    for i, sent in enumerate(sentences):
        print(f"   {i+1}. {sent[:80]}...")
    
    # Test extractive method directly
    print(f"\n🎯 Testing extractive method directly with max_length=400:")
    extractive_result = summarizer._enhanced_extractive_summarize(sentences, 400)
    print(f"📝 Extractive result:")
    print(f"   {extractive_result}")

if __name__ == "__main__":
    debug_ganesh_text()
