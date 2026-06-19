# Gestura Architecture

## System Overview

Gestura is a real-time sign language translation platform. It captures video frames from a mobile device camera, extracts pose/hand keypoints via MediaPipe Holistic, segments individual signs using motion analysis, classifies each sign with a deep learning model, and converts the ASL gloss sequence into grammatical English.

```
┌─────────────────────────────────────────────────────────────────────┐
│                          FRONTEND (Expo)                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────────────┐ │
│  │Onboarding│  │  Login   │  │Register  │  │ Camera (Translator) │ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────────────────┘ │
│                                                  │                  │
│                                     ┌────────────┼────────────┐     │
│                                     │ HTTP POST  │ WebSocket  │     │
│                                     │ /api/convert│ ws://...  │     │
│                                     └────────────┼────────────┘     │
└──────────────────────────────────────────────────┼──────────────────┘
                                                   │
┌──────────────────────────────────────────────────┼──────────────────┐
│                   API GATEWAY (Express/TS)       │                  │
│                                                  ▼                  │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                   Port 8080 (HTTP)                          │   │
│  │  POST /api/auth/register   → authRouter                     │   │
│  │  POST /api/auth/login      → authRouter                     │   │
│  │  POST /api/auth/logout     → authRouter                     │   │
│  │  GET  /api/auth/me         → authRouter  [JWT]              │   │
│  │  POST /api/convert         → translationRouter [JWT]        │   │
│  │  GET  /api/stop/:uuid      → translationRouter [JWT]        │   │
│  │  GET  /health              → public                         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                   Port 9898 (WebSocket)                      │   │
│  │  JWT auth via ?token query param                             │   │
│  │  Session tracking via ?uuid query param                      │   │
│  │  Pushes translation results on stop                          │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              MongoDB (mongoose / native driver)               │   │
│  │  gestura.users: { email, password(bcrypt), full_name, ... }  │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                    HTTP POST │ (internal)
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  TRANSLATION SERVICE (FastAPI/Python)               │
│                                                                      │
│  POST /process-frame   → Converter.point_detection()                │
│                        → MotionDetector.update()                    │
│                        → _predict_word()                            │
│                                                                      │
│  POST /stop            → grammar_fixer.fix_grammar()                 │
│                                                                      │
│  POST /translate       → Direct ML inference                        │
│  POST /convert-sentence→ Direct grammar correction                   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    PROCESSING PIPELINE                        │   │
│  │                                                              │   │
│  │  1. Frame arrives (base64 JPEG)                              │   │
│  │  2. MediaPipe Holistic → keypoints (258-dim)                 │   │
│  │  3. Bounded persistence (last-known hand fill-in)            │   │
│  │  4. Nose-normalization (translation invariance)              │   │
│  │  5. Exponential Moving Average (jitter smoothing)            │   │
│  │  6. MotionDetector (hysteresis-based boundary detection)     │   │
│  │  7. On boundary: normalize to 35-frame window                │   │
│  │  8. ML inference (BiLSTM+Transformer)                        │   │
│  │  9. Buffer word in session state                             │   │
│  │  10. On stop: FLAN-T5 grammar correction                     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Media: ML Model (best_model.keras, BiLSTM+Transformer)             │
│         MediaPipe Holistic (holistic_landmarker.task)                │
│         FLAN-T5 (google/flan-t5-small or fine-tuned variant)        │
└─────────────────────────────────────────────────────────────────────┘
                              │
                    HTTP POST │ (internal, dev/test)
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ML MODEL TRAINING (Python)                      │
│                                                                      │
│  model.py       → Architecture (BiLSTM + Transformer)               │
│  extract_data.py→ Video → MediaPipe → .npy keypoint files          │
│  DataAugmentor.py→ Rotation, scaling, jitter, mirror augmentation  │
│  DataNormalizer.py→ Keypoint normalization                          │
│  ModelTester.py  → Live webcam inference testing                    │
│                                                                      │
│  Data flow:                                                         │
│    Videos/ → extract_data.py → keypoint_data/ (35,258 .npy)        │
│                                → DataAugmentor (copy + augment)      │
│                                → keypoint_data_augmented/            │
│                                → model.py training                   │
│                                → best_model.keras + sign_classes.npy │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: Single Frame Processing

```
Camera Frame
    │
    ▼
[1] Frontend captures JPEG from react-native-vision-camera
    │
    ▼
[2] POST /api/convert (multipart: rawImage + uuid)
    │  JWT authentication via requireAuth middleware
    ▼
[3] API Gateway proxies to Translation Service
    │  POST /process-frame (base64-encoded image bytes)
    ▼
[4] Converter.point_detection()
    │  - Decode JPEG → RGB
    │  - MediaPipe Holistic Landmarker
    │  - Extract 258-dim feature vector:
    │    • Left hand:  21 landmarks × 3 (x,y,z) = 63
    │    • Right hand: 21 landmarks × 3 (x,y,z) = 63
    │    • Pose:       33 landmarks × 4 (x,y,z,vis) = 132
    │  - Handedness correction (swap L/R if only one hand visible)
    │  - Bounded persistence: fill in last-known hand positions
    │    for up to PERSIST_WINDOW (5) frames, then decay to zeros
    │  - Nose-normalization: subtract nose (landmark 0) xyz from all coords
    │  - Exponential Moving Average (alpha=0.4) for jitter smoothing
    ▼
[5] MotionDetector.update()
    │  - Compute raw motion: ||kp[t] - kp[t-1]||
    │  - EMA-smooth motion magnitude (alpha=0.6)
    │  - Hysteresis state machine with adaptive thresholds:
    │    • motion < stillness_floor (0.5) → still_counter++
    │    • motion < low_th (median * 0.5) → still_counter++
    │    • motion > high_th (median * 4.0) → still_counter = 0
    │  - Accumulate raw keypoints in sign buffer
    │  - When still_counter >= 8 frames → sign boundary detected
    ▼
[6] If sign boundary detected:
    │  - Normalize sign frames to 35-frame window (resample or pad)
    │  - ML inference: model.predict(normalized_window)
    │  - Map argmax index → word from sign_classes.npy
    │  - Skip "BACKGROUND" predictions (not real signs)
    │  - Append word to session's predicted_words list
    │  - Return { status: "word_detected", word: "...", sign_index: N }
    │
    │  If no boundary:
    │  - Return { status: "processing" }
    │
    │  If idle (both hands absent ≥ 15 frames):
    │  - Reset motion detector
    │  - Return { status: "idle" }
    ▼
[7] API Gateway returns response to frontend
```

## Data Flow: Session End

```
[1] User taps "Stop" / navigates away
    │
    ▼
[2] GET /api/stop/:uuid  (JWT auth)
    │
    ▼
[3] API Gateway proxies: POST /stop to Translation Service
    │
    ▼
[4] Translation Service:
    │  - Pops session state
    │  - Concatenates predicted_words → ASL gloss string
    │  - Runs FLAN-T5 grammar correction:
    │    "translate ASL gloss to English: <gloss>"
    │  - Returns { asl_gloss, english, words }
    ▼
[5] API Gateway:
    │  - Formats result as { type: "translation", asl_gloss, english, words }
    │  - Sends to client via WebSocket
    │  - Returns HTTP 200
    ▼
[6] Frontend:
    │  - Receives via WebSocket onmessage handler
    │  - Displays English translation
    │  - Optional: text-to-speech via expo-speech
    │  - Optional: haptic feedback via expo-haptics
```

---

## Service Boundaries & Responsibilities

### API Gateway (Express.js, TypeScript)
- **Port:** 8080 (HTTP), 9898 (WebSocket)
- **Role:** API gateway, auth provider, WebSocket relay
- **State:** Session UUID → WebSocket connection mapping (in-memory Map)
- **Data:** User records in MongoDB (email, bcrypt hash, metadata)
- **Dependencies:** MongoDB, Translation Service

### Translation Service (FastAPI, Python)
- **Port:** 7860 (HTTP, internal only)
- **Role:** Frame processing pipeline (segmentation + ML + grammar)
- **State:** Per-session accumulator (motion detector, predicted words)
- **Data:** ML model file, sign class mapping, MediaPipe model
- **Dependencies:** Keras/TensorFlow, Transformers, MediaPipe

### ML Model Training (Python/Keras)
- **No port** — standalone training/export
- **Output:** `best_model.keras` + `sign_classes.npy` → copied to Translation Service

### Frontend (Expo/React Native)
- **State:** JWT token (in-memory), session UUID (per camera screen lifecycle)
- **Auth:** Login/register/logout flows, token stored in memory
- **Camera:** react-native-vision-camera with frame capture loop
- **API:** HTTP for frame uploads, WebSocket for translation results

---

## Key Design Decisions

1. **Two-port API Gateway:** HTTP (8080) for REST, WebSocket (9898) for push — separate concerns, and the WebSocket server runs alongside Express without needing socket.io.

2. **Session state on Translation Service:** The MotionDetector and word buffer live entirely on the Python side. The API Gateway only tracks which UUID maps to which WebSocket connection. This avoids duplicating sign detection state across services.

3. **Bounded persistence for hands:** Rather than zeroing out when MediaPipe briefly loses a hand (common with fast motion), the system keeps the last-known position for up to 5 frames. This prevents false motion spikes during detection flicker.

4. **Hysteresis-based motion detection:** A simple state machine with adaptive thresholds works more reliably than fixed thresholds across different lighting conditions, camera distances, and signing speeds.

5. **Nose-normalization:** Subtracting the nose position from all keypoints makes the system translation-invariant — the signer can stand anywhere in the frame.

6. **Two-phase grammar correction:** ASL gloss words are accumulated during the session, and grammar correction runs once at session end via FLAN-T5. This avoids per-word latency and enables the model to use sentence-level context.

7. **BACKGROUND class:** The ML model explicitly predicts a "BACKGROUND" class for silence/noise periods. These predictions are filtered out by the segmentation pipeline before accumulation.

---

## Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| API Gateway | Express.js / TypeScript | ~4.x / ~5.x |
| Translation Service | FastAPI / Python | ~3.12 |
| ML Framework | Keras / TensorFlow | ~2.20 / ~2.20 |
| Grammar Correction | FLAN-T5 (HuggingFace) | google/flan-t5-small |
| Pose Estimation | MediaPipe Holistic | latest |
| Frontend | Expo / React Native | SDK 54 / 0.81 |
| Database | MongoDB | 7.0 |
| Containerization | Docker / Docker Compose | latest |
| Auth | bcrypt + JWT (jsonwebtoken) | bcrypt@5, jsonwebtoken@9 |
