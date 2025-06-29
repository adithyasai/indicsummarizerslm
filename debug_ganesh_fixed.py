#!/usr/bin/env python3
"""
Test the Ganesh Chaturthi text that was producing ugly truncations
"""

from proper_slm_summarizer import StateOfTheArtIndicSummarizer

def test_ganesh_issue():
    """Test with the specific Ganesh Chaturthi text that was being truncated badly"""
    
    ganesh_text = """గణేశ్ చతుర్థి అనేది హిందూ మతంలో అత్యంత ప్రాముఖ్యత కలిగిన పండుగలలో ఒకటి. ఈ పండుగను భగవంతుడు వినాయకుడి జన్మదినంగా ప్రతి సంవత్సరం భాద్రపద శుద్ధ చతుర్థి నాడు ఘనంగా జరుపుకుంటారు. గణపతి దేవుడు విజ్ఞానానికి, విజయానికి, ఆటంకాల నివారణకు ప్రతీకగా పూజించబడతాడు. ఈ పండుగ రోజున ఇంటిల్లిపాదీ గణపతి విగ్రహాన్ని స్థాపించి, పుష్పాలు, పళ్ళు, మోదకాలు (కుదురు లడ్డూలు) వంటి నైవేద్యాలతో గణేశుడిని భక్తితో పూజిస్తారు. దేశవ్యాప్తంగా, ముఖ్యంగా మహారాష్ట్ర, ఆంధ్రప్రదేశ్, తెలంగాణ, తమిళనాడు, కర్ణాటక వంటి రాష్ట్రాలలో ఈ పండుగను అత్యంత ఉత్సాహంగా జరుపుకుంటారు. పెద్ద పెద్ద పండుగ మండపాలలో భారీ విగ్రహాలను ప్రతిష్టించి, పలు రోజుల పాటు నృత్యాలు, సంగీత కార్యక్రమాలు, పూజలు నిర్వహిస్తారు. చివరగా, వినాయక నిమజ్జనంతో ఈ పండుగ ముగుస్తుంది. ఇది సామూహిక భక్తి, సాంస్కృతిక ఐక్యతను ప్రతిబింబించే పండుగగా మారింది. గణేశ్ చతుర్థి పండుగ సమయంలో పర్యావరణాన్ని దృష్టిలో ఉంచుకుని మట్టి విగ్రహాలను ప్రోత్సహించడం, పచ్చదనాన్ని కాపాడటం గురించి కూడా ప్రజల్లో చైతన్యం పెరుగుతోంది."""
    
    print("🧪 Testing Ganesh Chaturthi text that was getting truncated badly")
    print("=" * 70)
    print(f"Original text length: {len(ganesh_text)} characters")
    print()
    
    try:
        summarizer = StateOfTheArtIndicSummarizer()
        summary, method = summarizer.summarize(ganesh_text, max_length=300)
        
        print(f"📝 Generated Summary ({method}):")
        print(f"   {summary}")
        print()
        print(f"Summary length: {len(summary)} characters")
        
        # Check for ugly truncations - these should NOT be in the summary
        ugly_patterns = ["పచ్చద", "ము                       ుగుస్తుంది", "ప్రదేశ్                              "]
        
        has_ugly_truncation = any(pattern in summary for pattern in ugly_patterns)
        
        if has_ugly_truncation:
            print("❌ STILL HAS UGLY TRUNCATIONS!")
            for pattern in ugly_patterns:
                if pattern in summary:
                    print(f"   Found: '{pattern}'")
        else:
            print("✅ NO UGLY TRUNCATIONS! Summary looks clean.")
        
        # Check if summary ends properly
        if summary.endswith('.') or summary.endswith('।'):
            print("✅ Summary ends with proper punctuation")
        else:
            print(f"⚠️  Summary ends with: '{summary[-10:]}'")
        
        # Check if it's not just a copy
        if not ganesh_text.startswith(summary.replace("...", "").strip()):
            print("✅ Summary is not just a truncated copy")
        else:
            print("❌ Summary is still just a truncated copy")
            
        return summary, method
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

if __name__ == "__main__":
    test_ganesh_issue()
