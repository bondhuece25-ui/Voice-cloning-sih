import json

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from app.db.supabase import supabase
from app.ml.preprocessing import preprocess_audio
from app.services.model_service import predict
from app.services.risk_engine import TemporalRiskEngine


router = APIRouter()


TARGET_SAMPLE_RATE = 16000
INFERENCE_WINDOW_SECONDS = 1
INFERENCE_WINDOW_SAMPLES = (
    TARGET_SAMPLE_RATE * INFERENCE_WINDOW_SECONDS
)


@router.websocket("/ws/detect")
async def websocket_detection(websocket: WebSocket):

    await websocket.accept()

    try:
        # ---------------------------------------------------------
        # 1. AUTHENTICATION
        # ---------------------------------------------------------

        token = await websocket.receive_text()

        response = supabase.auth.get_user(token)

        if response.user is None:
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION
            )
            return

        user = response.user

        await websocket.send_json({
            "type": "connection",
            "status": "authenticated",
            "user_id": str(user.id)
        })

        # ---------------------------------------------------------
        # 2. RECEIVE AUDIO CONFIGURATION
        # ---------------------------------------------------------

        config_message = await websocket.receive_text()

        try:
            audio_config = json.loads(config_message)

        except json.JSONDecodeError:
            await websocket.close(
                code=status.WS_1003_UNSUPPORTED_DATA
            )
            return

        if audio_config.get("type") != "audio_config":
            await websocket.close(
                code=status.WS_1003_UNSUPPORTED_DATA
            )
            return

        sample_rate = audio_config.get("sample_rate")
        channels = audio_config.get("channels")
        audio_format = audio_config.get("format")

        if not isinstance(sample_rate, int):
            await websocket.close(
                code=status.WS_1003_UNSUPPORTED_DATA
            )
            return

        if sample_rate <= 0:
            await websocket.close(
                code=status.WS_1003_UNSUPPORTED_DATA
            )
            return

        if channels != 1:
            await websocket.close(
                code=status.WS_1003_UNSUPPORTED_DATA
            )
            return

        if audio_format != "pcm_s16le":
            await websocket.close(
                code=status.WS_1003_UNSUPPORTED_DATA
            )
            return

        await websocket.send_json({
            "type": "audio_config",
            "status": "accepted",
            "sample_rate": sample_rate,
            "channels": channels,
            "format": audio_format
        })

        print(
            f"WebSocket audio configured: "
            f"{sample_rate} Hz, "
            f"{channels} channel, "
            f"{audio_format}"
        )

        # ---------------------------------------------------------
        # 3. CREATE RISK ENGINE
        # ---------------------------------------------------------

        risk_engine = TemporalRiskEngine(window_size=5)

        # ---------------------------------------------------------
        # 4. CREATE AUDIO BUFFER
        # ---------------------------------------------------------

        audio_buffer = np.array(
            [],
            dtype=np.float32
        )

        print("Realtime ML detection started.")

        # ---------------------------------------------------------
        # 5. RECEIVE AUDIO CHUNKS
        # ---------------------------------------------------------

        while True:

            audio_chunk = await websocket.receive_bytes()

            if not audio_chunk:
                continue

            # -----------------------------------------------------
            # Convert PCM16 bytes → int16
            # -----------------------------------------------------

            pcm16_audio = np.frombuffer(
                audio_chunk,
                dtype=np.int16
            )

            if pcm16_audio.size == 0:
                continue

            # -----------------------------------------------------
            # Convert int16 → float32 [-1, 1]
            # -----------------------------------------------------

            waveform = (
                pcm16_audio.astype(np.float32)
                / 32768.0
            )

            # -----------------------------------------------------
            # Resample to 16 kHz
            # -----------------------------------------------------

            processed_audio, processed_sample_rate = (
                preprocess_audio(
                    waveform=waveform,
                    sample_rate=sample_rate
                )
            )

            if processed_sample_rate != TARGET_SAMPLE_RATE:
                raise ValueError(
                    "Audio was not converted to 16 kHz"
                )

            # -----------------------------------------------------
            # Add processed audio to buffer
            # -----------------------------------------------------

            audio_buffer = np.concatenate(
                (
                    audio_buffer,
                    processed_audio
                )
            )

            # -----------------------------------------------------
            # 6. RUN MODEL ONLY WHEN WE HAVE 1 SECOND
            # -----------------------------------------------------

            while len(audio_buffer) >= INFERENCE_WINDOW_SAMPLES:

                inference_audio = (
                    audio_buffer[
                        :INFERENCE_WINDOW_SAMPLES
                    ]
                )

                # Remove the processed window from buffer
                audio_buffer = (
                    audio_buffer[
                        INFERENCE_WINDOW_SAMPLES:
                    ]
                )

                print(
                    "Running ML inference on "
                    f"{len(inference_audio)} samples"
                )

                # -------------------------------------------------
                # 7. REAL MODEL INFERENCE
                # -------------------------------------------------

                score = predict(
                    inference_audio
                )

                # -------------------------------------------------
                # 8. TEMPORAL RISK ENGINE
                # -------------------------------------------------

                risk_result = risk_engine.add_score(
                    score
                )

                print(
                    f"Spoof probability: {score:.4f} | "
                    f"Average: "
                    f"{risk_result['average_score']:.4f} | "
                    f"Risk: "
                    f"{risk_result['risk_level']}"
                )

                # -------------------------------------------------
                # 9. SEND RESULT TO FRONTEND
                # -------------------------------------------------

                await websocket.send_json({
                    "type": "detection",

                    "current_score":
                        risk_result["current_score"],

                    "average_score":
                        risk_result["average_score"],

                    "risk_level":
                        risk_result["risk_level"],

                    "samples":
                        risk_result["samples"]
                })

    except WebSocketDisconnect:

        print(
            "WebSocket client disconnected."
        )

    except Exception as error:

        print(
            f"WebSocket error: {error}"
        )

        try:
            await websocket.close(
                code=status.WS_1011_INTERNAL_ERROR
            )

        except Exception:
            pass