from io import BytesIO
from uuid import uuid4

import numpy as np
import soundfile as sf

from fastapi import APIRouter, UploadFile, Depends, HTTPException

from app.core.auth import get_current_user
from app.db.supabase import supabase
from app.ml.preprocessing import preprocess_audio
from app.services.model_service import predict
from app.services.risk_engine import calculate_risk


router = APIRouter()

BUCKET_NAME = "audio"
TARGET_SAMPLE_RATE = 16000


# =========================================================
# POST /detect
# =========================================================

@router.post("/detect")
async def detect(
    file: UploadFile,
    user=Depends(get_current_user)
):
    try:
        # 1. Read uploaded audio
        audio_bytes = await file.read()

        if not audio_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded audio file is empty"
            )

        original_filename = file.filename or "audio.wav"

        # 2. Generate unique Storage path
        storage_path = (
            f"{user.id}/"
            f"{uuid4()}-"
            f"{original_filename}"
        )

        # 3. Upload original audio to Supabase Storage
        supabase.storage.from_(BUCKET_NAME).upload(
            path=storage_path,
            file=audio_bytes,
            file_options={
                "content-type": file.content_type or "audio/wav",
                "upsert": "false"
            }
        )

        # 4. Decode audio bytes into waveform
        try:
            waveform, sample_rate = sf.read(
                BytesIO(audio_bytes),
                dtype="float32",
                always_2d=True
            )

        except Exception as error:
            raise HTTPException(
                status_code=400,
                detail=f"Could not decode audio file: {error}"
            )

        if waveform.size == 0:
            raise HTTPException(
                status_code=400,
                detail="Audio file contains no samples"
            )

        # 5. Convert stereo/multi-channel audio to mono
        waveform = waveform.mean(
            axis=1,
            dtype=np.float32
        )

        # 6. Preprocess audio for Wav2Vec2
        processed_audio, processed_sample_rate = preprocess_audio(
            waveform=waveform,
            sample_rate=sample_rate,
            target_sample_rate=TARGET_SAMPLE_RATE
        )

        # 7. Run ML model
        score = predict(processed_audio)

        # 8. Calculate risk level
        risk = calculate_risk(score)

        # 9. Save detection metadata to database
        (
            supabase
            .table("detections")
            .insert({
                "user_id": str(user.id),
                "filename": original_filename,
                "storage_path": storage_path,
                "synthetic_probability": score,
                "risk_level": risk,
                "model_version": "wav2vec2-v1"
            })
            .execute()
        )

        # 10. Return detection result
        return {
            "user_id": str(user.id),
            "filename": original_filename,
            "storage_path": storage_path,
            "synthetic_probability": score,
            "risk_level": risk,
            "model_version": "wav2vec2-v1"
        }

    except HTTPException:
        raise

    except Exception as error:
        print(f"Detection error: {error}")

        raise HTTPException(
            status_code=500,
            detail="Detection failed"
        )


# =========================================================
# GET /history
# =========================================================

@router.get("/history")
async def detection_history(
    user=Depends(get_current_user)
):
    try:
        response = (
            supabase
            .table("detections")
            .select("*")
            .eq("user_id", str(user.id))
            .order("created_at", desc=True)
            .execute()
        )

        return {
            "detections": response.data
        }

    except Exception as error:
        print(f"History error: {error}")

        raise HTTPException(
            status_code=500,
            detail="Could not fetch detection history"
        )