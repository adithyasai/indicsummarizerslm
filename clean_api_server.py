#!/usr/bin/env python3
"""
Clean FastAPI server for Telugu Summarization
Uses the ProperSLMSummarizer with neural enhancement
"""

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn

# Import the summarizer
try:
    from proper_slm_summarizer import get_proper_slm_summarizer, ProperSLMSummarizer
    print("✅ Successfully imported ProperSLMSummarizer")
except Exception as e:
    print(f"❌ Failed to import summarizer: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Telugu Summarizer API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize summarizer (this will load the model)
print("🔄 Loading summarizer model...")
try:
    summarizer = get_proper_slm_summarizer()
    print("✅ Summarizer loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load summarizer: {e}")
    summarizer = None

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve a simple HTML form for testing"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Telugu Summarizer</title>
        <meta charset="UTF-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            textarea { width: 100%; padding: 10px; border: 1px solid #ccc; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }
            button:hover { background: #0056b3; }
            .result { margin-top: 20px; padding: 15px; background: #f8f9fa; border: 1px solid #dee2e6; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Telugu Summarizer</h1>
            <p>Enter Telugu text below to get an AI-generated summary:</p>
            
            <form action="/summarize" method="post">
                <textarea name="text" rows="8" placeholder="తెలుగు వచనం ఇక్కడ అతికించండి..." required></textarea><br><br>
                <button type="submit">📝 సారాంశం పొందండి (Get Summary)</button>
            </form>
            
            <div style="margin-top: 30px;">
                <h3>📋 Sample Telugu Text (for testing):</h3>
                <p style="background: #f0f0f0; padding: 10px; font-size: 14px;">
                G7 సమ్మిట్ అనేది ప్రపంచంలోని ఏడు ముఖ్యమైన పారిశ్రామిక అభివృద్ధి చెందిన దేశాల మధ్య జరిగే శిఖరాగ్ర సమావేశం. ఈ G7 దేశాలు – అమెరికా, కెనడా, బ్రిటన్, ఫ్రాన్స్, జర్మనీ, ఇటలీ మరియు జపాన్. ఇవి ప్రపంచ ఆర్థిక వ్యవస్థలో కీలకమైన ప్రాతినిధ్యం కలిగి ఉంటాయి.
                </p>
            </div>
        </div>
    </body>
    </html>
    """

@app.post("/summarize", response_class=HTMLResponse)
async def summarize_form(text: str = Form(...)):
    """Handle form submission and return HTML response"""
    if not summarizer:
        return """
        <html><body>
            <h3>❌ Error: Summarizer not loaded</h3>
            <a href="/">← Back</a>
        </body></html>
        """
    
    try:
        # Generate summary
        print(f"📝 Input text (length: {len(text)}): {text[:100]}...")
        summary, method = summarizer.summarize(text, max_length=200)
        print(f"📝 Generated summary (length: {len(summary)}): {summary}")
        print(f"🔧 Method used: {method}")
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Summary Result</title>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 800px; margin: 0 auto; }}
                .result {{ padding: 15px; background: #d4edda; border: 1px solid #c3e6cb; margin: 10px 0; }}
                .original {{ padding: 15px; background: #f8f9fa; border: 1px solid #dee2e6; margin: 10px 0; }}
                .stats {{ font-size: 14px; color: #666; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>✅ Summary Generated</h2>
                
                <div class="result">
                    <h3>📝 Summary:</h3>
                    <p>{summary}</p>
                </div>
                
                <div class="stats">
                    <strong>Method:</strong> {method}<br>
                    <strong>Original length:</strong> {len(text)} characters<br>
                    <strong>Summary length:</strong> {len(summary)} characters<br>
                    <strong>Compression ratio:</strong> {len(summary)/len(text)*100:.1f}%
                </div>
                
                <div class="original">
                    <h4>📄 Original Text:</h4>
                    <p style="font-size: 14px;">{text}</p>
                </div>
                
                <p><a href="/">← Generate Another Summary</a></p>
            </div>
        </body>
        </html>
        """
        
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        return f"""
        <html><body>
            <h3>❌ Error generating summary: {str(e)}</h3>
            <a href="/">← Back</a>
        </body></html>
        """

@app.post("/api/summarize")
async def summarize_api(request: Request):
    """API endpoint for JSON requests"""
    if not summarizer:
        raise HTTPException(status_code=500, detail="Summarizer not loaded")
    
    try:
        data = await request.json()
        text = data.get("text", "")
        max_length = data.get("max_length", 200)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text is required")
        
        print(f"🔗 API request - Text length: {len(text)}, Max length: {max_length}")
        print(f"🔗 Input text: {text[:100]}...")
        
        summary, method = summarizer.summarize(text, max_length=max_length)
        
        print(f"🔗 API result - Summary length: {len(summary)}, Method: {method}")
        print(f"🔗 Summary: {summary}")
        
        return JSONResponse({
            "summary": summary,
            "method": method,
            "original_length": len(text),
            "summary_length": len(summary),
            "compression_ratio": len(summary) / len(text) * 100
        })
        
    except Exception as e:
        logger.error(f"API summarization error: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = "healthy" if summarizer else "unhealthy"
    return {"status": status, "summarizer_loaded": summarizer is not None}

if __name__ == "__main__":
    print("🚀 Starting Telugu Summarizer API Server...")
    print("📍 Access the web interface at: http://localhost:8000")
    print("📍 API endpoint: POST http://localhost:8000/api/summarize")
    print("📍 Health check: GET http://localhost:8000/health")
    
    uvicorn.run(
        "clean_api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
