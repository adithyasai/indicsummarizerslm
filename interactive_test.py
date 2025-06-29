#!/usr/bin/env python3
"""
Interactive Telugu Summarizer Testing Tool - DEBUGGING VERSION
Test the improved summarizer with your own text inputs and debug issues
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

from proper_slm_summarizer import StateOfTheArtIndicSummarizer
from true_abstractive_summarizer import TrueAbstractiveSummarizer
import time

def interactive_testing():
    """Interactive testing interface for the Telugu summarizer with debugging"""
    
    print("🐛 INTERACTIVE TELUGU SUMMARIZER DEBUGGING")
    print("=" * 60)
    print("✨ Let's debug the summarizer issues!")
    print("📝 Enter Telugu text and see what's going wrong")
    print("🔧 We'll test with different settings to find the problem")
    print("❌ Type 'quit' or 'exit' to stop")
    print("-" * 60)
    
    # Initialize summarizer once
    print("🔄 Initializing summarizers...")
    try:
        old_summarizer = StateOfTheArtIndicSummarizer()
        new_summarizer = TrueAbstractiveSummarizer()
        print("✅ Both summarizers ready!")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    print("\n" + "="*60)
    
    test_count = 0
    
    while True:
        test_count += 1
        print(f"\n📝 TEST #{test_count}")
        print("-" * 30)
        
        # Get text input
        print("🔸 Enter Telugu text to summarize:")
        print("   (You can paste multi-line text, press Enter twice when done)")
        
        lines = []
        while True:
            try:
                line = input()
                if line.strip() == "":
                    if lines:  # If we have some content, break
                        break
                    else:  # If no content yet, continue
                        continue
                        
                if line.strip().lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    return
                    
                lines.append(line)
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                return
        
        text = "\n".join(lines).strip()
        
        if not text:
            print("⚠️  No text entered. Try again!")
            continue
        
        # Get max_length preference
        print(f"\n🔧 Choose summary length:")
        print("   1. Short (150 chars) - Usually 1-2 sentences")
        print("   2. Medium (250 chars) - Usually 2-3 sentences") 
        print("   3. Long (350 chars) - Usually 3-4 sentences")
        print("   4. Custom length")
        
        try:
            choice = input("👉 Enter choice (1-4) [default: 2]: ").strip()
            
            if choice == "1":
                max_length = 150
            elif choice == "3":
                max_length = 350
            elif choice == "4":
                custom = input("👉 Enter custom length (100-500): ").strip()
                max_length = max(100, min(500, int(custom))) if custom.isdigit() else 250
            else:  # Default or choice == "2"
                max_length = 250
                
        except ValueError:
            max_length = 250
            print("⚠️  Invalid input, using default length (250)")
        
        # Choose summarizer type
        print(f"\n🤖 Choose summarizer:")
        print("   1. OLD: State-of-the-art (copying sentences)")
        print("   2. NEW: True Abstractive (generates new text)")
        
        try:
            summ_choice = input("👉 Enter choice (1-2) [default: 2]: ").strip()
            use_new = summ_choice != "1"
        except:
            use_new = True
        
        # Process and display results
        print(f"\n⚡ Processing with {'NEW Abstractive' if use_new else 'OLD State-of-art'} (max_length={max_length})...")
        print("-" * 40)
        
        start_time = time.time()
        try:
            if use_new:
                summary = new_summarizer.create_true_summary(text, max_length=max_length)
                method = "true_abstractive"
                summary_sentences = [s.strip() for s in summary.split('.') if s.strip()]
            else:
                summary, method = old_summarizer.summarize(text, max_length=max_length)
                summary_sentences = old_summarizer._preprocess_text(summary)
                
            execution_time = time.time() - start_time
            
            # Display results
            print(f"✅ RESULTS:")
            print(f"   🕒 Time: {execution_time:.3f}s")
            print(f"   🔧 Method: {method}")
            print(f"   📏 Length: {len(summary)}/{max_length} characters")
            print(f"   📝 Sentences: {len(summary_sentences)}")
            print(f"   📊 Compression: {len(summary)/len(text):.1%}")
            
            print(f"\n🎯 SUMMARY:")
            print(f"   {summary}")
            
            # Show individual sentences if multiple
            if len(summary_sentences) > 1:
                print(f"\n📋 SENTENCE BREAKDOWN:")
                for i, sentence in enumerate(summary_sentences, 1):
                    print(f"   {i}. {sentence}")
            
        except Exception as e:
            print(f"❌ Summarization failed: {e}")
            continue
        
        print("\n" + "="*60)
        
        # Ask if user wants to continue
        continue_choice = input("🔄 Test another text? (y/n) [default: y]: ").strip().lower()
        if continue_choice in ['n', 'no']:
            print("👋 Thanks for testing!")
            break

def quick_demo():
    """Quick demo with sample texts"""
    
    print("🚀 QUICK DEMO MODE")
    print("-" * 30)
    
    sample_texts = [
        {
            "title": "Sample 1: Technology News",
            "text": """టెక్నాలజీ రంగంలో భారతదేశం వేగంగా అభివృద్ధి చెందుతోంది. కృత్రిమ మేధ, బ్లాక్‌చైన్, క్లౌడ్ కంప్యూటింగ్ వంటి ఆధునిక సాంకేతికతలలో దేశం ముందుండి నడుస్తోంది. హైదరాబాద్, బెంగళూరు, పుణె వంటి నగరాలు ప్రపంచ స్థాయి టెక్ హబ్‌లుగా మారాయి. మరో 10 సంవత్సరాలలో భారతదేశం ప్రపంచంలో టాప్ 3 టెక్నాలజీ దేశాలలో ఒకటిగా నిలుస్తుందని నిపుణులు అంచనా వేస్తున్నారు."""
        },
        {
            "title": "Sample 2: Health News", 
            "text": """కరోనా వైరస్ మహమారి తర్వాత ప్రజలలో ఆరోగ్య అవగాహన పెరిగింది. యోగా, వ్యాయామం, ఆరోగ్యకరమైన ఆహారం వైపు మరింత దృష్టి పెట్టుతున్నారు. ప్రభుత్వం కూడా ప్రాథమిక వైద్య సేవలను బలోపేతం చేస్తోంది. టెలిమెడిసిన్, డిజిటల్ హెల్త్ రికార్డ్స్ వంటి కొత్త పద్ధతులు ప్రవేశపెట్టబడుతున్నాయి."""
        }
    ]
    
    print("📝 Testing with sample texts...\n")
    
    old_summarizer = StateOfTheArtIndicSummarizer()
    new_summarizer = TrueAbstractiveSummarizer()
    
    for i, sample in enumerate(sample_texts, 1):
        print(f"🔸 {sample['title']}")
        print(f"📄 Original: {sample['text'][:100]}...")
        
        # Test both summarizers
        old_summary, old_method = old_summarizer.summarize(sample['text'], max_length=200)
        new_summary = new_summarizer.create_true_summary(sample['text'], max_length=200)
        
        print(f"❌ OLD Summary: {old_summary}")
        print(f"✅ NEW Summary: {new_summary}")
        print("-" * 50)

if __name__ == "__main__":
    print("🌟 TELUGU SUMMARIZER INTERACTIVE TESTER")
    print("=" * 50)
    print("Choose testing mode:")
    print("1. Interactive Testing (enter your own text)")
    print("2. Quick Demo (sample texts)")
    
    try:
        choice = input("👉 Enter choice (1-2) [default: 1]: ").strip()
        
        if choice == "2":
            quick_demo()
        else:
            interactive_testing()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
