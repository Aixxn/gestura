import os
from dotenv import load_dotenv

import numpy as np
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

from converter import Converter, FEATURE_DIM, WINDOW_SIZE
from motion_detector import MotionDetector

# Load .env overrides (FEATURE_DIM, WINDOW_SIZE)
load_dotenv()

# ---------------------------------------------------------------------------
# App & component initialisation
# ---------------------------------------------------------------------------

app = FastAPI(title="Sign Segmentation Service")

converter = Converter()
motion_detector = MotionDetector(
    low_factor=0.5,
    high_factor=4.0,
    still_frames_required=8,   # ~400 ms at 20 fps
    min_sign_duration=5,       # minimum sign length (noise filter)
    history_size=30,
    feature_dim=FEATURE_DIM,   # validated on every update()
    motion_smoothing=0.6,      # smooth motion signal to suppress jitter
    stillness_floor=0.3,       # motion below this is always "still"
)

# Per-session state (in production, replace with Redis)
session_states: dict = {}

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class FrameRequest(BaseModel):
    uuid: str
    image_bytes: str          # base64-encoded JPEG/PNG
    timestamp_ms: Optional[int] = None


class SignResult(BaseModel):
    sign_index: int
    keypoints_sequence: list  # list of per-frame 1662-dim vectors
    window: list              # sliding window – one (WINDOW_SIZE × FEATURE_DIM) frame
    start_frame: int
    end_frame: int
    motion_score_avg: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/process-frame")
async def process_frame(request: FrameRequest):
    """
    Process a single camera frame.

    Flow
    ----
    1. Decode base64 → raw image bytes
    2. **converter.point_detection(bytes)** → 1662-dim keypoint vector
       (the converter handles JPEG/PNG decoding internally – NO double decode)
    3. **converter.process_new_frame(kp)** → sliding window updated
    4. **motion_detector.update(kp)** → sign-boundary detection

    When a sign *ends*, a ``SignResult`` is returned immediately with the
    completed keypoint sequence and the current sliding window for the
    translation service.
    """
    try:
        # --- 1. raw bytes from base64 ---
        raw_bytes = base64.b64decode(request.image_bytes)

        # --- 2. extract keypoints (Converter does the cv2 decode) ---
        keypoints = converter.point_detection(raw_bytes)        # raw (zeros for undetected)
        persisted = converter.get_persisted_keypoints()          # motion-stable

        # --- 3. maintain sliding window buffer (raw keypoints → matches training) ---
        current_window = converter.process_new_frame(keypoints)

        # --- 4. sign-boundary detection (persisted → no zero-to-real spikes) ---
        sign_ended, completed_sign_keypoints = motion_detector.update(persisted, keypoints)

        # --- 5. session tracking ---
        session = session_states.setdefault(request.uuid, {
            "sign_count": 0,
            "total_frames": 0,
        })
        session["total_frames"] += 1

        # --- 6. respond ---
        if sign_ended and completed_sign_keypoints is not None:
            sign_index = session["sign_count"]
            session["sign_count"] += 1

            keypoints_list = [kp.tolist() for kp in completed_sign_keypoints]

            return SignResult(
                sign_index=sign_index,
                keypoints_sequence=keypoints_list,
                window=current_window.tolist(),
                start_frame=session["total_frames"] - len(completed_sign_keypoints),
                end_frame=session["total_frames"] - 1,
                motion_score_avg=0.01,          # TODO: track real motion
            )

        return {"status": "processing", "frame_processed": True}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/end-sequence")
async def end_sequence(uuid: str):
    """
    Manually end a signing sequence.

    Returns the final sliding window so the translation service can process
    any buffered frames, then cleans up the session.
    """
    if uuid in session_states:
        final_window = converter.stop()
        del session_states[uuid]
        return {
            "status": "sequence ended",
            "uuid": uuid,
            "final_window": final_window.tolist(),
        }
    return {"status": "no active session", "uuid": uuid}


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "sign-segmentation"}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
