import os
import uuid
import shutil
import asyncio
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from gradio_client import Client, handle_file
from dotenv import load_dotenv
from supabase import create_client, Client as SupabaseClient
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_KEY")

if not HF_TOKEN:
    logger.error("HF_TOKEN not found in environment variables")
    raise ValueError("HF_TOKEN is required")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    logger.error("SUPABASE_URL or SUPABASE_SERVICE_KEY not found in environment variables")
    raise ValueError("Supabase credentials are required")

app = FastAPI(title="AI Image Generator", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global clients
ghibli_client = None
upscaler_client = None
supabase: SupabaseClient = None

@app.on_event("startup")
async def startup_event():
    global ghibli_client, upscaler_client, supabase
    try:
        logger.info("Initializing Ghibli Gradio client...")
        ghibli_client = Client("Han-123/EasyControl_Ghibli", hf_token=HF_TOKEN)
        logger.info("Ghibli Gradio client initialized successfully")

        logger.info("Initializing Upscaler Gradio client...")
        upscaler_client = Client("themaisk/themaisk-image-upscaler", hf_token=HF_TOKEN)
        logger.info("Upscaler Gradio client initialized successfully")
        
        logger.info("Initializing Supabase client...")
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        logger.info("Supabase client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize clients: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "ghibli_client_ready": ghibli_client is not None,
        "upscaler_client_ready": upscaler_client is not None,
        "supabase_ready": supabase is not None
    }

@app.post("/generate/")
async def generate_image(
    file: UploadFile = File(...),
    uid: str = Form(...)
):
    """Generate Ghibli-style image from input image, upscale it, and save to Supabase"""
    temp_input_path = None
    temp_generated_path = None
    temp_upscaled_path = None

    try:
        # Improved image validation
        content_type = file.content_type or ""
        filename = file.filename or ""

        # Check content type OR file extension
        valid_content_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']

        is_valid_content_type = any(content_type.startswith(ct) for ct in ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/'])
        is_valid_extension = any(filename.lower().endswith(ext) for ext in valid_extensions)

        if not (is_valid_content_type or is_valid_extension):
            logger.warning(f"Invalid file - Content-Type: {content_type}, Filename: {filename}")
            raise HTTPException(status_code=400, detail="File must be an image (jpg, png, webp)")

        if not uid or len(uid.strip()) == 0:
            raise HTTPException(status_code=400, detail="UID cannot be empty")

        logger.info(f"Starting image generation for user {uid}")
        logger.info(f"File info - Content-Type: {content_type}, Filename: {filename}")

        # Create temp directory if it doesn't exist
        temp_dir = Path("/tmp")
        temp_dir.mkdir(exist_ok=True)

        # Save uploaded image temporarily
        image_id = str(uuid.uuid4())
        file_extension = Path(filename).suffix or '.jpg'
        temp_input_path = temp_dir / f"{image_id}_input{file_extension}"

        # Save file
        with open(temp_input_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        logger.info(f"Image saved to {temp_input_path}")

        # Validate file size (optional)
        file_size = temp_input_path.stat().st_size
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")

        # Check if clients are available
        if ghibli_client is None:
            raise HTTPException(status_code=503, detail="AI image generation service not available")
        
        if upscaler_client is None:
            raise HTTPException(status_code=503, detail="AI upscaler service not available")

        if supabase is None:
            raise HTTPException(status_code=503, detail="Storage service not available")

        # Step 1: Generate Ghibli-style image
        logger.info("Generating Ghibli-style image...")
        generated_result = await asyncio.wait_for(
            asyncio.to_thread(_predict_ghibli, str(temp_input_path)),
            timeout=300.0  # 5 minutes timeout
        )

        if not generated_result:
            raise HTTPException(status_code=500, detail="Invalid response from Ghibli AI model")

        # Extract the generated image path
        generated_image_path = generated_result.get("path") if isinstance(generated_result, dict) else generated_result
        logger.info(f"Ghibli image generated: {generated_image_path}")

        # Step 2: Upscale the generated image
        logger.info("Upscaling generated image...")
        upscaled_result = await asyncio.wait_for(
            asyncio.to_thread(_predict_upscale, generated_image_path),
            timeout=300.0  # 5 minutes timeout
        )

        if not upscaled_result:
            raise HTTPException(status_code=500, detail="Invalid response from upscaler AI model")

        logger.info(f"Image upscaled: {upscaled_result}")

        # Step 3: Upload upscaled image to Supabase avatars bucket
        avatar_url = await _upload_avatar_to_supabase(upscaled_result, uid)
        logger.info(f"Avatar uploaded to Supabase: {avatar_url}")

        # Step 4: Save dpurl to Firestore
        await _save_dpurl_to_firestore(uid, avatar_url)

        return JSONResponse({
            "success": True,
            "avatar_url": avatar_url,
            "uid": uid
        })

    except asyncio.TimeoutError:
        logger.error("Image generation/upscaling timed out after 5 minutes")
        raise HTTPException(
            status_code=408, 
            detail="Image generation/upscaling timed out. Please try again."
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except Exception as e:
        logger.error(f"Error generating image: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate image: {str(e)}"
        )

    finally:
        # Cleanup temporary files
        for temp_path in [temp_input_path, temp_generated_path, temp_upscaled_path]:
            if temp_path and Path(temp_path).exists():
                try:
                    Path(temp_path).unlink()
                    logger.info(f"Cleaned up temp file: {temp_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file {temp_path}: {e}")

async def _upload_avatar_to_supabase(local_image_path: str, uid: str) -> str:
    """Upload avatar image to Supabase storage and return public URL"""
    try:
        image_path = Path(local_image_path)
        if not image_path.exists():
            raise Exception(f"Image file not found: {local_image_path}")

        # Generate unique filename for Supabase storage
        image_id = str(uuid.uuid4())
        storage_path = f"{uid}/{image_id}.png"

        # Read image file
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()

        logger.info(f"Uploading avatar to Supabase: {storage_path}")

        # Upload to Supabase storage (avatars bucket)
        try:
            result = supabase.storage.from_("avatars").upload(
                path=storage_path,
                file=image_data,
                file_options={
                    "content-type": "image/png",
                    "cache-control": "3600"
                }
            )
            logger.info(f"Upload result: {result}")

        except Exception as upload_error:
            logger.error(f"Upload failed: {upload_error}")
            raise Exception(f"Supabase upload failed: {upload_error}")

        # Get public URL
        try:
            url_result = supabase.storage.from_("avatars").get_public_url(storage_path)
            logger.info(f"Generated public URL: {url_result}")

            if not url_result:
                raise Exception("Failed to get public URL")

            return url_result

        except Exception as url_error:
            logger.error(f"Failed to get public URL: {url_error}")
            raise Exception(f"Failed to get public URL: {url_error}")

    except Exception as e:
        logger.error(f"Failed to upload avatar to Supabase: {e}")
        raise Exception(f"Storage upload failed: {str(e)}")

async def _save_dpurl_to_firestore(uid: str, dpurl: str):
    """Save dpurl to Firestore user document"""
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        # Initialize Firebase Admin if not already done
        if not firebase_admin._apps:
            try:
                # Use the specified service account file path
                cred = credentials.Certificate("/etc/secrets/services")
                firebase_admin.initialize_app(cred)
            except Exception as e:
                logger.error(f"Failed to initialize Firebase with service account: {e}")
                raise Exception("Firebase initialization failed")

        db = firestore.client()

        logger.info(f"Saving dpurl to Firestore for user: {uid}")

        # Update user document with dpurl
        user_ref = db.collection("users").document(uid)
        
        # Update or create the dpurl field
        user_ref.set({
            "dpurl": dpurl
        }, merge=True)
        
        logger.info(f"Successfully saved dpurl for user {uid}")

    except Exception as e:
        logger.error(f"Failed to save dpurl to Firestore: {e}", exc_info=True)
        # Don't raise exception here - image generation was successful
        # Just log the error and continue

def _predict_ghibli(image_path: str):
    """Synchronous function to call the Ghibli Gradio client"""
    try:
        return ghibli_client.predict(
            prompt="Ghibli Studio style, Charming hand-drawn anime-style illustration",
            spatial_img=handle_file(image_path),
            height=768,
            width=768,
            seed=42,
            control_type="Ghibli",
            api_name="/single_condition_generate_image"
        )
    except Exception as e:
        logger.error(f"Ghibli Gradio client prediction failed: {e}")
        raise

def _predict_upscale(image_path: str):
    """Synchronous function to call the Upscaler Gradio client"""
    try:
        logger.info(f"Upscaling image: {image_path}")
        
        # Use fn_index=1 with RealESRGAN_x4plus_anime_6B model
        result = upscaler_client.predict(
            image_path,  # Input image
            "RealESRGAN_x4plus_anime_6B",  # Upscaler model (anime 6B)
            0,  # Denoise Strength
            True,  # Face Enhancement (GFPGAN)
            1,  # Resolution upscale
            fn_index=1
        )
        
        logger.info(f"Upscaling result: {result}")
        return result  # Return the upscaled image path
        
    except Exception as e:
        logger.error(f"Upscaler Gradio client prediction failed: {e}")
        raise

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception handler caught: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        timeout_keep_alive=300,  # 5 minutes keep alive
        timeout_graceful_shutdown=30
    )
