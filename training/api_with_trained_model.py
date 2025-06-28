"""
AadiShakthiSLM - Enhanced API with Trained Model
Updated API server with the newly trained Telugu/Hindi SLM model
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List
import logging
import os
import torch
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
from production_telugu_summarizer import get_production_telugu_summarizer
from neural_telugu_summarizer import get_neural_telugu_summarizer
from simple_telugu_summarizer import get_simple_telugu_summarizer
from trained_model_summarizer import get_trained_model_summarizer
from proper_slm_summarizer import get_proper_slm_summarizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AadiShakthiSLM API - Enhanced",
    description="Small Language Model for Telugu and Hindi with trained model",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if os.path.exists("ui"):
    app.mount("/static", StaticFiles(directory="ui"), name="static")

# Request/Response models
class SummarizeRequest(BaseModel):
    text: str
    language: Optional[str] = "auto"
    max_length: Optional[int] = 150
    min_length: Optional[int] = 30

class SummarizeResponse(BaseModel):
    summary: str
    original_length: int
    summary_length: int
    language_detected: str
    confidence: float
    model_used: str

class GenerateRequest(BaseModel):
    prompt: str
    language: Optional[str] = "auto"
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.7

class GenerateResponse(BaseModel):
    generated_text: str
    prompt: str
    language_detected: str
    model_used: str

# Language detection
from proper_slm_summarizer import get_proper_slm_summarizer

# Unified language detection using the Proper SLM Summarizer's methods
summarizer_for_lang = get_proper_slm_summarizer()
def detect_language(text: str) -> tuple[str, float]:
    """Detect language using SLM's robust detection"""
    lang_code = summarizer_for_lang.detect_language(text)
    _, confidence = summarizer_for_lang.get_language_confidence(text)
    # Map to API language labels
    mapping = {'te': 'telugu', 'hi': 'hindi'}
    language = mapping.get(lang_code, 'english')
    return language, confidence

# Model loading
trained_model = None
trained_tokenizer = None

def load_trained_model():
    """Load the trained model"""
    global trained_model, trained_tokenizer
    
    try:
        model_path = Path("models/final_model_summarization_te")
        
        if not model_path.exists():
            logger.warning("No trained model checkpoint found")
            return False
        
        # Load tokenizer
        try:
            trained_tokenizer = GPT2Tokenizer.from_pretrained(str(model_path))
        except:
            trained_tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        
        if trained_tokenizer.pad_token is None:
            trained_tokenizer.pad_token = trained_tokenizer.eos_token
        
        # Load model
        try:
            trained_model = GPT2LMHeadModel.from_pretrained(str(model_path))
        except:
            trained_model = AutoModelForCausalLM.from_pretrained(str(model_path))
        
        trained_model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trained_model.to(device)
        
        logger.info(f"✅ Trained model loaded successfully on {device}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load trained model: {e}")
        return False

# Initialize trained model
model_loaded = load_trained_model()

# Initialize Telugu summarizer (proper SLM version following GitHub documentation)
try:
    telugu_summarizer = get_proper_slm_summarizer()
    logger.info("✅ Proper SLM Telugu Summarizer initialized successfully (following documentation)")
except Exception as e:
    logger.warning(f"Could not initialize proper SLM summarizer: {e}. Trying trained model fallback.")
    try:
        telugu_summarizer = get_trained_model_summarizer()
        logger.info("✅ Trained Model Telugu Summarizer initialized as fallback")
    except Exception as e2:
        logger.warning(f"Could not initialize trained model summarizer: {e2}. Trying simple fallback.")
        try:
            telugu_summarizer = get_simple_telugu_summarizer()
            logger.info("✅ Simple Telugu Summarizer initialized as fallback")
        except Exception as e3:
            logger.warning(f"Could not initialize simple Telugu summarizer: {e3}. Trying production fallback.")
            try:
                telugu_summarizer = get_production_telugu_summarizer()
                logger.info("✅ Production Telugu Summarizer initialized as final fallback")
            except Exception as e4:
                logger.error(f"Could not initialize any Telugu summarizer: {e4}")
                telugu_summarizer = None

# Keep old enhanced summarizer as backup
try:
    from enhanced_summarizer import get_enhanced_summarizer
    enhanced_summarizer = get_enhanced_summarizer()
    logger.info("✅ Enhanced Telugu/Hindi Summarizer also available as backup")
except ImportError as e:
    logger.warning(f"Could not import enhanced summarizer: {e}. Using basic fallback.")
    enhanced_summarizer = None

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main UI"""    # Determine status
    has_proper_slm = telugu_summarizer and hasattr(telugu_summarizer, '__class__') and 'ProperSLM' in telugu_summarizer.__class__.__name__
    status_class = "success" if has_proper_slm else "warning"
    
    if has_proper_slm:
        model_status = "✅ Proper SLM Summarizer Active (Following GitHub Documentation)"
    elif model_loaded:
        model_status = "⚠️ Custom Trained Model Loaded (Fallback Mode)"
    else:
        model_status = "⚠️ Using Basic Fallback Model"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AadiShakthiSLM - Enhanced API</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
            .header {{ text-align: center; color: #333; margin-bottom: 30px; }}
            .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
            textarea {{ width: 100%; height: 100px; padding: 10px; border: 1px solid #ccc; border-radius: 5px; }}
            button {{ background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }}
            button:hover {{ background: #0056b3; }}
            .result {{ margin-top: 15px; padding: 15px; background: #f8f9fa; border-radius: 5px; }}
            .status {{ padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .success {{ background: #d4edda; color: #155724; }}
            .warning {{ background: #fff3cd; color: #856404; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 AadiShakthiSLM - Enhanced API</h1>
                <p>Telugu & Hindi Small Language Model with Custom Training</p>
            </div>
            
            <div class="status {status_class}">
                <strong>Model Status:</strong> {model_status}
            </div>
            
            <div class="section">
                <h2>📝 Text Summarization</h2>
                <textarea id="summaryText" placeholder="Enter Telugu or Hindi text to summarize..."></textarea>
                <br><br>
                <button onclick="summarizeText()">Summarize</button>
                <div id="summaryResult" class="result" style="display:none;"></div>
            </div>
            
            <div class="section">
                <h2>✨ Text Generation</h2>
                <textarea id="generatePrompt" placeholder="Enter a prompt for text generation..."></textarea>
                <br><br>
                <button onclick="generateText()">Generate</button>
                <div id="generateResult" class="result" style="display:none;"></div>
            </div>
            
            <div class="section">
                <h2>📊 API Endpoints</h2>
                <ul>
                    <li><strong>POST /summarize</strong> - Text summarization</li>
                    <li><strong>POST /generate</strong> - Text generation</li>
                    <li><strong>GET /health</strong> - Health check</li>
                    <li><strong>GET /model-info</strong> - Model information</li>
                </ul>
            </div>
        </div>
          <script>
            async function summarizeText() {{
                const text = document.getElementById('summaryText').value;
                if (!text) {{
                    alert('Please enter some text to summarize');
                    return;
                }}
                
                try {{
                    // Show loading message
                    document.getElementById('summaryResult').innerHTML = '<em>Processing...</em>';
                    document.getElementById('summaryResult').style.display = 'block';
                    
                    const response = await fetch('/summarize', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ text: text }})
                    }});
                    
                    if (!response.ok) {{
                        const errorText = await response.text();
                        throw new Error(`HTTP error! status: ${{response.status}} - ${{errorText}}`);
                    }}
                    
                    const result = await response.json();
                    document.getElementById('summaryResult').innerHTML = 
                        `<strong>Summary:</strong> ${{result.summary || 'No summary generated'}}<br>
                         <strong>Language:</strong> ${{result.language_detected || 'Unknown'}}<br>
                         <strong>Model:</strong> ${{result.model_used || 'Unknown'}}<br>
                         <strong>Compression:</strong> ${{result.original_length || 0}} → ${{result.summary_length || 0}} chars`;
                }} catch (error) {{
                    console.error('Summarization error:', error);
                    document.getElementById('summaryResult').innerHTML = 
                        `<strong>Error:</strong> ${{error.message}}<br>Please try again with different text.`;
                }}
                document.getElementById('summaryResult').style.display = 'block';
            }}
            
            async function generateText() {{
                const prompt = document.getElementById('generatePrompt').value;
                if (!prompt) {{
                    alert('Please enter a prompt for text generation');
                    return;
                }}
                
                try {{
                    // Show loading message
                    document.getElementById('generateResult').innerHTML = '<em>Processing...</em>';
                    document.getElementById('generateResult').style.display = 'block';
                    
                    const response = await fetch('/generate', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ prompt: prompt }})
                    }});
                    
                    if (!response.ok) {{
                        const errorText = await response.text();
                        throw new Error(`HTTP error! status: ${{response.status}} - ${{errorText}}`);
                    }}
                    
                    const result = await response.json();
                    document.getElementById('generateResult').innerHTML = 
                        `<strong>Generated:</strong> ${{result.generated_text || 'No text generated'}}<br>
                         <strong>Language:</strong> ${{result.language_detected || 'Unknown'}}<br>
                         <strong>Model:</strong> ${{result.model_used || 'Unknown'}}`;
                }} catch (error) {{
                    console.error('Generation error:', error);
                    document.getElementById('generateResult').innerHTML = 
                        `<strong>Error:</strong> ${{error.message}}<br>Please try again with a different prompt.`;
                }}
                document.getElementById('generateResult').style.display = 'block';
            }}
        </script>
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_text(request: SummarizeRequest):
    """Summarize text using the enhanced Telugu summarizer"""
    try:
        # Detect language
        language, confidence = detect_language(request.text)
        
        # Use enhanced Telugu summarizer first
        if telugu_summarizer and language in ["telugu", "hindi"]:
            try:
                summary, method = telugu_summarizer.summarize(request.text, request.max_length)
                
                return SummarizeResponse(
                    summary=summary,
                    original_length=len(request.text),
                    summary_length=len(summary),
                    language_detected=language,
                    confidence=confidence,
                    model_used=f"Enhanced Telugu Summarizer ({method})"
                )
            except Exception as e:
                logger.warning(f"Telugu summarizer failed: {e}, trying other methods")
        
        # Use enhanced summarizer if available (backup)
        if enhanced_summarizer:
            try:
                result = enhanced_summarizer.summarize(
                    request.text, 
                    max_length=request.max_length,
                    language=language
                )
                
                return SummarizeResponse(
                    summary=result.get('summary', ''),
                    original_length=len(request.text),
                    summary_length=len(result.get('summary', '')),
                    language_detected=language,
                    confidence=confidence,                    model_used="Enhanced Backup Summarizer"
                )
            except Exception as e:
                logger.warning(f"Enhanced summarizer failed: {e}, using basic method")
        
        # Fallback: Use trained model directly
        if model_loaded and trained_model and trained_tokenizer:
            try:
                # Use trained model with better prompting
                if language == "telugu":
                    prompt = f"ఈ వార్తను సంక్షిప్తంగా వ్రాయండి:\n{request.text}\n\nసారాంశం:"
                elif language == "hindi":
                    prompt = f"इस समाचार का संक्षिप्त सारांश लिखें:\n{request.text}\n\nसारांश:"
                else:
                    prompt = f"Summarize this text:\n{request.text}\n\nSummary:"
                
                # Limit input length
                max_input_length = 800
                if len(prompt) > max_input_length:
                    # Keep the prompt structure but truncate text
                    text_limit = max_input_length - 100  # Leave space for prompt
                    truncated_text = request.text[:text_limit] + "..."
                    if language == "telugu":
                        prompt = f"ఈ వార్తను సంక్షిప్తంగా వ్రాయండి:\n{truncated_text}\n\nసారాంశం:"
                    elif language == "hindi":
                        prompt = f"इस समाचार का संक्षिप्त सारांश लिखें:\n{truncated_text}\n\nसारांश:"
                    else:
                        prompt = f"Summarize this text:\n{truncated_text}\n\nSummary:"
                
                inputs = trained_tokenizer.encode(prompt, return_tensors="pt", max_length=1000, truncation=True)
                
                if inputs.shape[1] == 0:
                    raise ValueError("Empty input after tokenization")
                
                with torch.no_grad():
                    outputs = trained_model.generate(
                        inputs,
                        max_length=min(inputs.shape[1] + request.max_length, 1024),
                        min_length=inputs.shape[1] + 20,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=trained_tokenizer.eos_token_id,
                        eos_token_id=trained_tokenizer.eos_token_id,
                        repetition_penalty=1.2,
                        num_return_sequences=1
                    )
                
                generated_text = trained_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract summary
                if "సారాంశం:" in generated_text:
                    summary = generated_text.split("సారాంశం:")[-1].strip()
                elif "सारांश:" in generated_text:
                    summary = generated_text.split("सारांश:")[-1].strip()
                elif "Summary:" in generated_text:
                    summary = generated_text.split("Summary:")[-1].strip()
                else:
                    summary = generated_text[len(prompt):].strip()
                
                # Clean summary
                summary = summary.replace('\n', ' ').strip()
                
                # Fallback if summary is too short
                if len(summary) < 20:
                    raise ValueError("Generated summary too short")
                
                model_used = "Custom Trained SLM (Direct)"
                
            except Exception as model_error:
                logger.warning(f"Model generation failed: {model_error}, using extractive fallback")
                # Simple extractive fallback
                sentences = request.text.split('.')
                sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
                summary = '. '.join(sentences[:3]) + "." if len(sentences) >= 3 else request.text[:request.max_length]
                model_used = "Extractive Fallback"
        else:
            # Basic extractive summarization
            sentences = request.text.split('.')
            sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
            summary = '. '.join(sentences[:3]) + "." if len(sentences) >= 3 else request.text[:request.max_length]
            model_used = "Basic Extractive"
        
        return SummarizeResponse(
            summary=summary,
            original_length=len(request.text),
            summary_length=len(summary),
            language_detected=language,
            confidence=confidence,
            model_used=model_used
        )
        
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        # Final fallback
        first_sentences = request.text.split('.')[:2]
        fallback_summary = '. '.join(first_sentences) + "."
        
        return SummarizeResponse(
            summary=fallback_summary,
            original_length=len(request.text),
            summary_length=len(fallback_summary),
            language_detected="unknown",
            confidence=0.0,
            model_used="Emergency Fallback"
        )

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text using the trained model"""
    try:
        # Detect language
        language, confidence = detect_language(request.prompt)
        
        if model_loaded and trained_model and trained_tokenizer:
            try:
                # Use trained model with better error handling
                prompt = request.prompt
                
                # Limit input length
                max_input_length = 512
                if len(prompt) > max_input_length:
                    prompt = prompt[:max_input_length]
                
                inputs = trained_tokenizer.encode(prompt, return_tensors="pt", max_length=max_input_length, truncation=True)
                
                if inputs.shape[1] == 0:
                    raise ValueError("Empty input after tokenization")
                
                with torch.no_grad():
                    outputs = trained_model.generate(
                        inputs,
                        max_length=min(inputs.shape[1] + request.max_length, 1024),
                        temperature=request.temperature,
                        do_sample=True,
                        pad_token_id=trained_tokenizer.eos_token_id,
                        eos_token_id=trained_tokenizer.eos_token_id,
                        num_return_sequences=1
                    )
                
                generated_text = trained_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Safe extraction of generated part
                if len(generated_text) > len(prompt):
                    generated_part = generated_text[len(prompt):].strip()
                else:
                    generated_part = generated_text.strip()
                
                # Fallback if generation is empty
                if not generated_part:
                    generated_part = "Generated text is empty. Please try a different prompt."
                
                model_used = "Custom Trained SLM"
                
            except Exception as model_error:
                logger.warning(f"Model generation failed: {model_error}, using fallback")
                generated_part = f"Model generation failed. Error: {str(model_error)}"
                model_used = "Error Fallback"
        else:
            # Simple fallback
            generated_part = "I apologize, but the trained model is not available. Please try again later."
            model_used = "Fallback Response"
        
        return GenerateResponse(
            generated_text=generated_part,
            prompt=request.prompt,
            language_detected=language,
            model_used=model_used
        )
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return GenerateResponse(
            generated_text=f"Error: {str(e)}",
            prompt=request.prompt,
            language_detected="unknown",
            model_used="Error Handler"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "model_type": "Custom Trained SLM" if model_loaded else "Fallback",
        "torch_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

@app.get("/model-info")
async def model_info():
    """Get model information"""
    info = {
        "model_loaded": model_loaded,
        "model_type": "Custom Trained SLM" if model_loaded else "Fallback",
        "supports_languages": ["Telugu", "Hindi", "English"],
        "capabilities": ["Summarization", "Text Generation"],
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    if model_loaded and trained_model:
        info["model_parameters"] = sum(p.numel() for p in trained_model.parameters())
        info["tokenizer_vocab_size"] = trained_tokenizer.vocab_size if trained_tokenizer else 0
    
    return info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
