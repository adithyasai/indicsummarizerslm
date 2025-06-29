#!/usr/bin/env python3
"""
Debug neural model outputs to see what's going wrong
"""

from proper_slm_summarizer import StateOfTheArtIndicSummarizer

def debug_neural_models():
    """Test neural models directly to see raw outputs"""
    
    text = """భారతదేశం అనేది ప్రపంచంలో అత్యంత వైవిధ్యభరిత దేశాలలో ఒకటిగా గుర్తించబడింది. భాష, సంస్కృతి, మతం, వేషభాష, ఆచారాలు, భోజన అలవాట్లు మొదలైన అనేక అంశాలలో భారతదేశంలో విభిన్నత స్పష్టంగా కనిపిస్తుంది."""
    
    print("🔍 DEBUGGING NEURAL MODEL OUTPUTS")
    print("=" * 60)
    print(f"Input text: {text}")
    print()
    
    summarizer = StateOfTheArtIndicSummarizer()
    
    # Test IndicBART directly
    print("🤖 Testing IndicBART:")
    try:
        indicbart_result = summarizer._indicbart_summarize(text, 200)
        print(f"   Raw output: {repr(indicbart_result)}")
        print(f"   Clean output: {indicbart_result}")
        print(f"   Length: {len(indicbart_result)}")
        
        # Character analysis
        telugu_chars = sum(1 for char in indicbart_result if 0x0C00 <= ord(char) <= 0x0C7F)
        chinese_chars = sum(1 for char in indicbart_result if 0x4E00 <= ord(char) <= 0x9FFF)
        print(f"   Telugu chars: {telugu_chars}")
        print(f"   Chinese chars: {chinese_chars}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    
    # Test mT5 directly 
    print("🤖 Testing mT5:")
    try:
        mt5_result = summarizer._mt5_summarize(text, 200)
        print(f"   Raw output: {repr(mt5_result)}")
        print(f"   Clean output: {mt5_result}")
        print(f"   Length: {len(mt5_result)}")
        
        # Character analysis
        telugu_chars = sum(1 for char in mt5_result if 0x0C00 <= ord(char) <= 0x0C7F)
        special_tokens = mt5_result.count("<extra_id")
        print(f"   Telugu chars: {telugu_chars}")
        print(f"   Special tokens: {special_tokens}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    debug_neural_models()
