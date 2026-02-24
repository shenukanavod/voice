"""
Voice Authentication REST API
FastAPI-based REST API for voice verification
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
import logging
from datetime import datetime
from scipy.spatial.distance import cosine
import librosa

# Import existing modules
import model_loader
import audio_utils
try:
    from liveness_detection import LivenessDetector
except ImportError:
    LivenessDetector = None
from attempt_limiter import AttemptLimiter, format_lockout_message

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Voice Authentication API",
    description="REST API for voice-based user authentication",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
EMBEDDINGS_DIR = Path("embeddings")
MODEL_PATH = Path("models/speaker_embedding_model.pth")
LOGS_DIR = Path("logs/api")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Global model cache (load once at startup)
_model_cache = None
_liveness_detector = None
_attempt_limiter = None


def get_model():
    """Load model once and cache it."""
    global _model_cache
    if _model_cache is None:
        logger.info("Loading speaker embedding model...")
        _model_cache = model_loader.load_model(model_path=str(MODEL_PATH), verbose=False)
        logger.info("Model loaded and cached")
    return _model_cache


def get_liveness_detector():
    """Load liveness detector once and cache it."""
    global _liveness_detector
    if _liveness_detector is None:
        if LivenessDetector is None:
            logger.warning("Liveness detection module not available")
            return None
        logger.info("Initializing liveness detector...")
        _liveness_detector = LivenessDetector()
        logger.info("Liveness detector initialized")
    return _liveness_detector


def get_attempt_limiter():
    """Load attempt limiter once and cache it."""
    global _attempt_limiter
    if _attempt_limiter is None:
        logger.info("Initializing attempt limiter...")
        _attempt_limiter = AttemptLimiter()
        logger.info("Attempt limiter initialized")
    return _attempt_limiter


# Response models
class VerificationResponse(BaseModel):
    """Response model for verification endpoint."""
    verified: bool = Field(..., description="Whether user is verified")
    user_id: str = Field(..., description="User identifier")
    similarity_score: float = Field(..., description="Voice similarity score (0-1)")
    threshold: float = Field(..., description="Threshold used for verification")
    is_live: bool = Field(..., description="Whether audio passed liveness detection")
    liveness_confidence: float = Field(..., description="Liveness detection confidence (0-1)")
    timestamp: str = Field(..., description="Verification timestamp (ISO format)")
    message: str = Field(..., description="Result message")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(..., description="Error timestamp")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    embeddings_available: int
    timestamp: str


def calculate_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculate cosine similarity between embeddings."""
    return float(1 - cosine(emb1, emb2))


def log_verification_attempt(user_id: str, verified: bool, similarity: Optional[float], 
                             threshold: float, liveness_result: Optional[Dict[str, Any]] = None, 
                             error: Optional[str] = None, attempt_info: Optional[Dict[str, Any]] = None):
    """Log verification attempt to file."""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'user_id': user_id,
        'verified': verified,
        'similarity_score': float(similarity) if similarity else None,
        'threshold': float(threshold),
        'liveness_check': liveness_result if liveness_result else None,
        'error': error,
        'attempt_info': attempt_info
    }
    
    log_file = LOGS_DIR / f"api_verifications_{datetime.now().strftime('%Y%m%d')}.json"
    
    # Append to log
    import json
    logs = []
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                logs = json.load(f)
        except Exception:
            logs = []
    
    logs.append(log_entry)
    
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("Starting Voice Authentication API...")
    
    # Verify model exists
    if not MODEL_PATH.exists():
        logger.error(f"Model file not found: {MODEL_PATH}")
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    
    # Verify embeddings directory exists
    if not EMBEDDINGS_DIR.exists():
        logger.warning(f"Embeddings directory not found: {EMBEDDINGS_DIR}")
        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Pre-load model
    try:
        get_model()
        logger.info("API startup complete")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Voice Authentication API",
        "version": "1.0.0",
        "endpoints": {
            "verify": "/verify (POST)",
            "health": "/health (GET)",
            "docs": "/docs (GET)",
            "redoc": "/redoc (GET)"
        },
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        model = get_model()
        model_loaded = model is not None
        
        # Count enrolled users
        enrolled_users = len(list(EMBEDDINGS_DIR.glob("*.npy"))) if EMBEDDINGS_DIR.exists() else 0
        
        return HealthResponse(
            status="healthy",
            model_loaded=model_loaded,
            embeddings_available=enrolled_users,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            embeddings_available=0,
            timestamp=datetime.now().isoformat()
        )


@app.post("/verify", response_model=VerificationResponse)
async def verify_voice(
    user_id: str = Form(..., description="User ID to verify"),
    audio: UploadFile = File(..., description="Audio file (WAV or FLAC)"),
    threshold: float = Form(0.85, description="Similarity threshold (0.0-1.0)", ge=0.0, le=1.0)
):
    """
    Verify user identity using voice biometrics.
    
    **Parameters:**
    - **user_id**: User identifier (must be enrolled)
    - **audio**: Audio file in WAV or FLAC format (recommended: 3+ seconds)
    - **threshold**: Similarity threshold for verification (default: 0.85)
    
    **Returns:**
    - **verified**: Boolean indicating if user is verified
    - **similarity_score**: Cosine similarity score (0-1)
    - **threshold**: Threshold used
    - **message**: Human-readable result message
    
    **Example:**
    ```bash
    curl -X POST "http://localhost:8000/verify" \\
         -F "user_id=john_doe" \\
         -F "audio=@test_audio.wav" \\
         -F "threshold=0.85"
    ```
    """
    temp_audio_path = None
    
    try:
        # Initialize attempt limiter
        limiter = get_attempt_limiter()
        
        # Check if user is locked out
        is_locked, remaining = limiter.is_locked(user_id)
        if is_locked:
            error_msg = format_lockout_message(remaining if remaining is not None else 0.0)
            logger.warning(f"User {user_id} locked out, {remaining:.1f}s remaining")
            attempt_info = limiter.get_status(user_id)
            log_verification_attempt(user_id, False, None, threshold, None, error_msg, attempt_info)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "message": error_msg,
                    "locked_until": attempt_info.get('remaining_lockout_seconds'),
                    "failed_attempts": attempt_info.get('failed_attempts')
                }
            )
        
        # Validate audio file extension
        if not audio.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Audio filename is required"
            )
        
        file_extension = Path(audio.filename).suffix.lower()
        if file_extension not in ['.wav', '.flac']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported audio format: {file_extension}. Use .wav or .flac"
            )
        
        logger.info(f"Verification request: user_id={user_id}, filename={audio.filename}")
        
        # Check if user is enrolled
        embedding_file = EMBEDDINGS_DIR / f"{user_id}.npy"
        if not embedding_file.exists():
            error_msg = f"User '{user_id}' not enrolled"
            logger.warning(error_msg)
            attempt_result = limiter.record_failure(user_id)
            log_verification_attempt(user_id, False, None, threshold, None, error_msg, attempt_result)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User '{user_id}' not enrolled. Please enroll the user first."
            )
        
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_audio_path = Path(temp_file.name)
            shutil.copyfileobj(audio.file, temp_file)
        
        logger.info(f"Audio saved to temporary file: {temp_audio_path}")
        
        # Load enrolled embedding
        try:
            enrolled_embedding = np.load(embedding_file)
            logger.info(f"Loaded enrolled embedding for {user_id}")
        except Exception as e:
            error_msg = f"Failed to load enrolled embedding: {e}"
            logger.error(error_msg)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_msg
            )
        
        # Load model
        try:
            model = get_model()
        except Exception as e:
            error_msg = f"Failed to load model: {e}"
            logger.error(error_msg)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_msg
            )
        
        # LIVENESS DETECTION - Critical security check
        logger.info("Running liveness detection...")
        try:
            # Load audio for liveness check
            audio_data, sr = librosa.load(temp_audio_path, sr=16000)
            
            # Run liveness detection
            detector = get_liveness_detector()
            if detector is None:
                # Fallback if liveness detector is not available
                is_live = True
                liveness_result = {'confidence': 1.0, 'passed_checks': ['fallback'], 'checks': ['fallback'], 'message': 'Liveness detection not available'}
            else:
                is_live, liveness_result = detector.is_live_voice(audio_data, sr)
            
            logger.info(
                f"Liveness check: is_live={is_live}, "
                f"confidence={liveness_result['confidence']:.2%}, "
                f"passed={len(liveness_result['passed_checks'])}/{len(liveness_result['checks'])}"
            )
            
            # REJECT if not live - immediate security rejection
            if not is_live:
                error_msg = f"Liveness check failed: {liveness_result['message']}"
                logger.warning(f"Authentication denied for {user_id}: {error_msg}")
                attempt_result = limiter.record_failure(user_id)
                log_verification_attempt(user_id, False, 0.0, threshold, liveness_result, error_msg, attempt_result)
                
                return VerificationResponse(
                    verified=False,
                    user_id=user_id,
                    similarity_score=0.0,
                    threshold=threshold,
                    is_live=False,
                    liveness_confidence=liveness_result['confidence'],
                    timestamp=datetime.now().isoformat(),
                    message=f"Authentication denied: {error_msg}. Remaining attempts: {attempt_result['remaining_attempts']}"
                )
        
        except Exception as e:
            error_msg = f"Liveness detection failed: {e}"
            logger.error(error_msg)
            # In case of liveness detection failure, allow fallback (or reject for strict security)
            # For strict security, uncomment the following:
            # raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg)
            logger.warning("Continuing without liveness check due to error")
            is_live = True  # Fallback
            liveness_result = {'confidence': 0.5, 'message': f'Liveness check error: {e}'}
        
        # Process audio and extract embedding
        try:
            tensor = audio_utils.prepare_audio_for_model(
                temp_audio_path,
                sample_rate=16000,
                n_mels=128,
                duration=3.0,
                device='cpu',
                check_quality=True
            )
            
            with torch.no_grad():
                test_embedding = model(tensor)
            
            test_embedding_np = test_embedding.cpu().numpy().squeeze()
            logger.info("Test embedding extracted successfully")
            
        except FileNotFoundError as e:
            error_msg = f"Audio file error: {e}"
            logger.error(error_msg)
            attempt_result = limiter.record_failure(user_id)
            log_verification_attempt(user_id, False, None, threshold, None, error_msg, attempt_result)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )
        except ValueError as e:
            error_msg = f"Invalid audio: {e}"
            logger.error(error_msg)
            attempt_result = limiter.record_failure(user_id)
            log_verification_attempt(user_id, False, None, threshold, None, error_msg, attempt_result)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )
        except Exception as e:
            error_msg = f"Audio processing failed: {e}"
            logger.error(error_msg)
            attempt_result = limiter.record_failure(user_id)
            log_verification_attempt(user_id, False, None, threshold, None, error_msg, attempt_result)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_msg
            )
        
        # Calculate similarity
        try:
            similarity = calculate_similarity(enrolled_embedding, test_embedding_np)
            logger.info(f"Similarity calculated: {similarity:.4f}")
        except Exception as e:
            error_msg = f"Similarity calculation failed: {e}"
            logger.error(error_msg)
            attempt_result = limiter.record_failure(user_id)
            log_verification_attempt(user_id, False, None, threshold, None, error_msg, attempt_result)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_msg
            )
        
        # Verification decision
        is_verified = similarity >= threshold
        
        # Handle attempt limiting based on verification result
        if is_verified:
            # Success - reset attempt counter
            limiter.record_success(user_id)
            attempt_info = limiter.get_status(user_id)
            message = f"Access granted - Voice verified (similarity: {similarity:.2%}, liveness: {liveness_result['confidence']:.2%})"
        else:
            # Failure - increment counter
            attempt_result = limiter.record_failure(user_id)
            attempt_info = attempt_result
            remaining_attempts = attempt_result['remaining_attempts']
            if remaining_attempts > 0:
                message = f"Access denied - Similarity too low ({similarity:.2%} < {threshold:.0%}). {remaining_attempts} attempts remaining."
            else:
                message = f"Access denied - Account locked for {attempt_result['lockout_duration']} seconds due to too many failed attempts."
        
        # Log successful verification attempt with liveness data and attempt info
        log_verification_attempt(user_id, is_verified, similarity, threshold, liveness_result, None, attempt_info)
        
        logger.info(f"Verification result: user={user_id}, verified={is_verified}, similarity={similarity:.4f}, liveness={liveness_result['confidence']:.2%}")
        
        # Return response with liveness information
        return VerificationResponse(
            verified=is_verified,
            user_id=user_id,
            similarity_score=round(similarity, 4),
            threshold=threshold,
            is_live=is_live,
            liveness_confidence=round(liveness_result['confidence'], 4),
            timestamp=datetime.now().isoformat(),
            message=message
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        # Catch any unexpected errors
        error_msg = f"Unexpected error during verification: {str(e)}"
        logger.error(error_msg, exc_info=True)
        try:
            limiter = get_attempt_limiter()
            attempt_result = limiter.record_failure(user_id)
            log_verification_attempt(user_id, False, None, threshold, None, error_msg, attempt_result)
        except Exception:
            log_verification_attempt(user_id, False, None, threshold, None, error_msg, None)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )
        
    finally:
        # Clean up temporary file
        if temp_audio_path and temp_audio_path.exists():
            try:
                temp_audio_path.unlink()
                logger.info(f"Cleaned up temporary file: {temp_audio_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {e}")


@app.get("/users", response_model=dict)
async def list_enrolled_users():
    """
    List all enrolled users.
    
    **Returns:**
    - List of enrolled user IDs
    - Total count
    """
    try:
        if not EMBEDDINGS_DIR.exists():
            return {
                "users": [],
                "count": 0,
                "message": "No users enrolled"
            }
        
        users = [f.stem for f in EMBEDDINGS_DIR.glob("*.npy")]
        
        return {
            "users": sorted(users),
            "count": len(users),
            "message": f"Found {len(users)} enrolled user(s)"
        }
        
    except Exception as e:
        logger.error(f"Failed to list users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list users: {e}"
        )


@app.get("/users/{user_id}", response_model=dict)
async def check_user_enrollment(user_id: str):
    """
    Check if a specific user is enrolled.
    
    **Parameters:**
    - **user_id**: User identifier to check
    
    **Returns:**
    - Enrollment status
    - Embedding file path (if exists)
    """
    embedding_file = EMBEDDINGS_DIR / f"{user_id}.npy"
    is_enrolled = embedding_file.exists()
    
    if is_enrolled:
        try:
            # Load embedding to get info
            embedding = np.load(embedding_file)
            return {
                "user_id": user_id,
                "enrolled": True,
                "embedding_file": str(embedding_file),
                "embedding_dimension": len(embedding),
                "message": f"User '{user_id}' is enrolled"
            }
        except Exception as e:
            return {
                "user_id": user_id,
                "enrolled": True,
                "embedding_file": str(embedding_file),
                "error": f"Embedding file exists but failed to load: {e}",
                "message": f"User '{user_id}' enrollment file corrupted"
            }
    else:
        return {
            "user_id": user_id,
            "enrolled": False,
            "message": f"User '{user_id}' is not enrolled"
        }


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("Voice Authentication REST API")
    print("=" * 70)
    print()
    print("Starting server...")
    print(f"Model: {MODEL_PATH}")
    print(f"Embeddings: {EMBEDDINGS_DIR}")
    print()
    print("API will be available at:")
    print("  - http://localhost:8000")
    print("  - Interactive docs: http://localhost:8000/docs")
    print("  - ReDoc: http://localhost:8000/redoc")
    print()
    print("Endpoints:")
    print("  POST /verify          - Verify user voice")
    print("  GET  /health          - Health check")
    print("  GET  /users           - List enrolled users")
    print("  GET  /users/{user_id} - Check user enrollment")
    print()
    print("=" * 70)
    print()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
