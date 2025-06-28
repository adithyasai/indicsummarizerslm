"""
AadiShakthiSLM - Small Language Model API for Telugu and Hindi
Main API server with text summarization capabilities
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AadiShakthiSLM API",
    description="Small Language Model for Telugu and Hindi text summarization",
    version="1.0.0"
)

# Enable CORS for all origins (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for UI
app.mount("/static", StaticFiles(directory="ui"), name="static")

# Request/Response models
class SummarizeRequest(BaseModel):
    text: str
    language: Optional[str] = "auto"  # "auto", "telugu", "hindi"
    max_length: Optional[int] = 150
    min_length: Optional[int] = 30

class SummarizeResponse(BaseModel):
    summary: str
    original_length: int
    summary_length: int
    language_detected: str
    confidence: float

# Simple language detection (placeholder)
def detect_language(text: str) -> tuple[str, float]:
    """Simple language detection based on script"""
    # Telugu script range: \u0C00-\u0C7F
    # Hindi/Devanagari script range: \u0900-\u097F
    
    telugu_chars = sum(1 for char in text if '\u0C00' <= char <= '\u0C7F')
    hindi_chars = sum(1 for char in text if '\u0900' <= char <= '\u097F')
    total_chars = len([char for char in text if char.isalpha()])
    
    if total_chars == 0:
        return "unknown", 0.0
    
    telugu_ratio = telugu_chars / total_chars
    hindi_ratio = hindi_chars / total_chars
    
    if telugu_ratio > hindi_ratio and telugu_ratio > 0.5:
        return "telugu", telugu_ratio
    elif hindi_ratio > 0.5:
        return "hindi", hindi_ratio
    else:
        return "english", max(0.5, 1 - telugu_ratio - hindi_ratio)

# Initialize the enhanced SLM summarizer with custom tokenizer
try:
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.enhanced_slm_api import get_summarizer
    summarizer = get_summarizer()
    logger.info("✅ Enhanced SLM with custom IndicTokenizer initialized successfully")
    
    # Keep fallback imports
    try:
        from src.enhanced_summarizer import enhance_summarization
        logger.info("Enhanced summarizer also available as backup")
    except ImportError:
        enhance_summarization = None
        
except ImportError as e:
    logger.warning(f"Could not import enhanced SLM: {e}. Using basic fallback.")
    try:
        from src.enhanced_summarizer import enhance_summarization
        logger.info("Enhanced summarizer loaded as fallback")
    except ImportError:
        enhance_summarization = None
        logger.warning("Enhanced summarizer also unavailable, using basic summarization")
    summarizer = None

# Simple extractive summarization (fallback method)
def simple_extractive_summarize(text: str, max_length: int = 150) -> str:
    """Simple extractive summarization with improved sentence scoring"""
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return text[:max_length] + "..." if len(text) > max_length else text
    
    # Score sentences by position, length, and keyword density
    scored_sentences = []
    for i, sentence in enumerate(sentences):
        words = sentence.split()
        
        # Position score (earlier sentences get higher scores)
        position_score = 1.0 / (i + 1)
        
        # Length score (prefer sentences with reasonable length)
        length_score = min(len(words) / 15.0, 1.0) if len(words) > 3 else 0.1
        
        # Keyword density (simple heuristic)
        keyword_score = len([w for w in words if len(w) > 4]) / max(len(words), 1)
        
        total_score = position_score * 0.5 + length_score * 0.3 + keyword_score * 0.2
        scored_sentences.append((sentence, total_score))
    
    # Sort by score and select top sentences
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    
    summary = ""
    for sentence, _ in scored_sentences:
        if len(summary) + len(sentence) + 2 <= max_length:
            summary += sentence + ". "
        else:
            break
    
    return summary.strip() or sentences[0] + "."

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main UI page"""
    try:
        with open("ui/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
            <body>
                <h1>AadiShakthiSLM API</h1>
                <p>API is running! UI files not found.</p>
                <p>Visit <a href="/docs">/docs</a> for API documentation</p>
            </body>
        </html>
        """)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "AadiShakthiSLM API is running"}

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_text(request: SummarizeRequest):
    """
    Summarize text in Telugu, Hindi, or English using enhanced SLM with custom tokenizer
    """
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Detect language if auto
        if request.language == "auto":
            detected_lang, confidence = detect_language(request.text)
        else:
            detected_lang = request.language
            confidence = 1.0
        
        # Use enhanced SLM summarizer with custom tokenizer
        if summarizer is not None:
            try:
                result = summarizer.summarize(
                    text=request.text,
                    max_length=request.max_length,
                    method="auto",  # Will use abstractive if model trained, else enhanced extractive
                    language=detected_lang
                )
                
                summary = result["summary"]
                detected_lang = result["language"]
                confidence = result["confidence"]
                method_used = result["method"]
                
                logger.info(f"✅ Enhanced SLM used - Method: {method_used}, "
                           f"Language: {detected_lang}, Confidence: {confidence:.2f}")
                
            except Exception as e:
                logger.warning(f"Enhanced SLM failed: {e}. Using fallback.")
                summary = simple_extractive_summarize(request.text, request.max_length)
        
        # Fallback to enhanced summarization
        elif enhance_summarization is not None:
            try:
                result = enhance_summarization(request.text, max_sentences=3)
                summary = result["summary"]
                detected_lang = result["language"]
                confidence = 0.8  # Good confidence for enhanced method
                
                logger.info(f"Enhanced summarization fallback - Method: {result['method']}, "
                           f"Compression: {result['compression_ratio']:.2f}")
                
            except Exception as e:
                logger.warning(f"Enhanced summarization failed: {e}. Using basic fallback.")
                summary = simple_extractive_summarize(request.text, request.max_length)
            except Exception as e:
                logger.warning(f"SLM summarization failed: {e}. Using basic fallback.")
                summary = simple_extractive_summarize(request.text, request.max_length)
        
        # Basic fallback
        else:
            summary = simple_extractive_summarize(request.text, request.max_length)
        
        # Ensure minimum length is respected
        if len(summary.split()) < request.min_length // 5:  # Rough word count
            summary = request.text[:request.max_length] + "..."
        
        response = SummarizeResponse(
            summary=summary,
            original_length=len(request.text),
            summary_length=len(summary),
            language_detected=detected_lang,
            confidence=confidence
        )
        
        logger.info(f"Summarized text: {len(request.text)} -> {len(summary)} chars, "
                   f"Language: {detected_lang}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in summarization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

@app.get("/languages")
async def get_supported_languages():
    """Get list of supported languages"""
    return {
        "supported_languages": [
            {"code": "auto", "name": "Auto-detect"},
            {"code": "telugu", "name": "Telugu"},
            {"code": "hindi", "name": "Hindi"},
            {"code": "english", "name": "English"}
        ]
    }

@app.get("/model/info")
async def get_model_info():
    """Get information about the current model"""
    return {
        "model_name": "AadiShakthiSLM-v1.0",
        "version": "1.0.0",
        "supported_languages": ["Telugu", "Hindi", "English"],
        "max_input_length": 2048,
        "model_type": "extractive_summarization",
        "status": "basic_implementation"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
