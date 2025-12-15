"""
TrueSight API Server
====================
Exposes the TrueSight Engine as a REST API.
"""

import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from core.truesight_engine import TrueSightEngine, EngineConfig

# Global Engine Instance
engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    global engine
    print("🚀 API Starting up... Loading TrueSight Engine...")
    # Use default config (Giant model, tuned thresholds)
    try:
        engine = TrueSightEngine()
        print("✅ TrueSight Engine Ready.")
    except Exception as e:
        print(f"❌ Failed to load engine: {e}")
        raise e
    yield
    print("🛑 API Shutting down...")
    # Clean up resources if needed

app = FastAPI(title="TrueSight API", version="1.0", lifespan=lifespan)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if engine is None:
        return JSONResponse(status_code=503, content={"status": "Engine not ready"})
    return {"status": "online", "device": engine.device, "model": engine.config.dino_model_size}

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    """
    Analyze an uploaded video file.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    # Validate file type
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    # Save temp file
    # Note: For production, consider streaming or processing directly if supported by decord
    # Decord needs a file path usually.
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        print(f"📥 Processing upload: {file.filename} -> {tmp_path}")
        
        # Run Analysis
        # We assume run_rppg=False by default for speed, unless specified (can add query param)
        result = engine.analyze(tmp_path, run_rppg=False)
        
        # Clean up
        os.unlink(tmp_path)
        
        return result.to_dict()

    except Exception as e:
        # Cleanup if failed
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        print(f"❌ Error analyzing {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Run access from any IP, port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
