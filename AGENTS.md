# Agent Instructions for Gestura

## Project Structure: Microservices + Expo Mobile

**Gestura** is a gesture recognition platform with:
- **Frontend**: Expo React Native app (camera capture, sign language recognition)
- **Backend Services**: Multiple TypeScript/Python microservices + Python ML pipeline
- **Services**: API Gateway, Authentication Service, Translation Service, Sign Segmentation Service, ML Model
- **Infrastructure**: Docker Compose (Redis for session storage)

### Directory Layout

```
apiGateway/          → Express.js API gateway (Node.js, TypeScript)
frontend/            → Expo app with file-based routing (React Native)
translationService/  → FastAPI service for ASL-to-English translation (Python)
signSegmentationService/ → Service for real-time sign boundary detection (Python)
model/               → ML model training & inference (Python + Keras)
authenticationService/ → Currently empty/placeholder
```

## Frontend (Expo React Native)

**Tech**: Expo 54, React 19, React Native 0.81, file-based routing via expo-router

### Development
```bash
cd frontend
npm install
npx expo start  # Opens dev menu
# Options: press 'a' for Android emulator, 'i' for iOS simulator, 'w' for web
```

**Key Config** (`app.json`):
- API endpoints hardcoded in `extra`: `apiUrl`, `wsUrl` — **must be updated for IP** before deployment
- Requires camera, audio, internet permissions (Android)
- Firebase config in `.env` (preconfigured)

**Common Tasks**:
- **Lint**: `npm run lint`
- **Build for Android**: `npm run android` (requires Android Studio)
- **Build for iOS**: `npm run ios` (macOS only)
- **Web**: `npm run web`

**Gotchas**:
- Camera & vision features use `react-native-vision-camera` v4.7.2 + `react-native-fast-opencv`
- WebSocket URLs in `app.json` must match backend host/port (WS_PORT=9898 by default)

---

## API Gateway (Express.js + TypeScript)

**Tech**: Express, TypeScript, tsconfig-paths for `@src/*` aliases, Mocha for tests

### Development
```bash
cd apiGateway
npm install
npm run dev         # Start with nodemon + ts-node (dev server)
npm run build       # Compile to dist/, run tsc-alias
npm start           # Run compiled dist/bin/www.js (production)
npm test            # Run Mocha tests (timeout 5000ms)
```

**Build Workflow**:
1. TypeScript → JavaScript (tsc)
2. Path alias resolution (tsc-alias)
3. Output to `dist/` directory

**Key Files**:
- `bin/www.ts` → Server entry point
- `app.ts` → Express setup
- `routes/` → API endpoints
- `tsconfig.json` → Includes `baseUrl: "."`, paths: `@src/*` → `./`

**Environment** (`.env`):
```
WEB_SOCKET_HOST = "0.0.0.0"
WEB_SOCKET_PORT = 9898
SEGMENTATION_SERVICE_URL = "http://signsegmentationservice:8000"
```

**Gotchas**:
- Uses `tsc-alias` for path resolution post-compilation (required for `@src/*` imports)
- Strict TypeScript (`strict: true`, `noUnusedLocals`, etc.)
- WebSocket server on separate port 9898 from API (8080)

---

## Translation Service (FastAPI + Python)

**Tech**: FastAPI, Keras/TensorFlow, Groq LLM API (for ASL grammar correction)

### Development
```bash
cd translationService
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

**Key Functionality**:
- Accepts batch of sign sequences and returns translated words
- Uses Groq LLM API (llama-3.3-70b) to fix grammar
- FastAPI debug mode enabled in `main.py`

**Environment** (`.env`):
- `GROQ_API_KEY` required for LLM integration

**Port**: 7860 (FastAPI default)

---

## Sign Segmentation Service (Python + MediaPipe)

**Tech**: Python 3.9, FastAPI, MediaPipe for keypoint extraction, hysteresis-based motion detection

### Development
```bash
cd signSegmentationService
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Key Functionality**:
- Receives video frames via HTTP POST
- Extracts keypoints using MediaPipe Holistic
- Tracks hand motion with hysteresis and adaptive thresholding
- Detects sign boundaries when motion falls below threshold for N frames
- Emits sign-level results (keypoint sequence for each detected sign)
- Buffers signs internally until a sign ends, then returns the completed sign

**Key Components**:
- `main.py` → FastAPI server with endpoints
- `motion_detector.py` → Core logic with hysteresis and adaptive threshold
- `converter.py` → MediaPipe keypoint extraction (shared with legacy code)

**Endpoints**:
- `POST /process-frame` → Process a single frame, returns sign data when a sign ends
- `GET /health` → Health check

**Environment**: No special environment variables required

**Port**: 8000 (FastAPI default)

---

## ML Model (Python + Keras/TensorFlow)

**Tech**: Keras/TensorFlow 2.20, MediaPipe, LSTM/Transformer architecture

### Files
- `model.keras` / `best_model.keras` → Pre-trained weights (~41MB each)
- `model.py` → Model architecture + training
- `test_main.py` → Tests

### Development
```bash
cd model
pip install -r requirements.txt
python main.py          # Run inference/training
pytest test_main.py     # Run tests
```

**Key Components**:
- `model.py` → LSTM/Transformer for gesture classification
- `DataAugmentor.py` → Data augmentation pipeline
- `DataNormalizer.py` → Keypoint normalization
- `Selected_extraction.py` → Feature selection from MediaPipe
- `modelVisualizer.py` → Visualization utils

**Data**:
- `keypoint_data/` → Raw extracted keypoints
- `keypoint_data_augmented/` → Augmented dataset
- `keypoint_data_normalized/` → Normalized keypoints

**Gotchas**:
- Model files are large (41MB) — not committed; checkpoint restoration required
- MediaPipe expects 17+ keypoints per frame
- Training on augmented data; normalization required for consistency

---

## Infrastructure: Docker Compose

**Services**:
- `apigateway` → Node.js server, ports 8080 (API), 9898 (WebSocket)
- `translationService` → Python FastAPI service (port 7860)
- `signSegmentationService` → Python FastAPI service (port 8000)
- `redis` → Cache/session store (port 6379)

### Local Development
```bash
docker-compose up -d
docker-compose logs -f apigateway
docker-compose down
```

**Redis Setup**:
- Used for session storage in API Gateway (optional, can be in-memory)
- Single instance, no replication needed for MVP

**Image Names** (for CI/CD):
- `baronocasiones/gestura-apigateway:dev`
- `baronocasiones/gestura-translationservice:dev`
- `baronocasiones/gestura-signsegmentationservice:dev`

---

## Common Commands by Service

### API Gateway
```bash
cd apiGateway
npm run dev         # Development (watch mode)
npm run build       # TypeScript → dist/
npm test            # Mocha tests
npm start           # Production
```

### Frontend
```bash
cd frontend
npx expo start      # Dev server with menu
npm run lint        # ESLint check
npm run android     # Build for Android
```

### Translation Service
```bash
cd translationService
uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

### Sign Segmentation Service
```bash
cd signSegmentationService
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Model
```bash
cd model
python main.py
pytest test_main.py
```

---

## Required Environment Variables

### API Gateway (`.env`)
```
WEB_SOCKET_HOST=0.0.0.0
WEB_SOCKET_PORT=9898
SEGMENTATION_SERVICE_URL=http://signsegmentationservice:8000
```

### Frontend (`.env`)
```
FIREBASE_API_KEY=...
FIREBASE_AUTH_DOMAIN=...
FIREBASE_PROJECT_ID=...
FIREBASE_STORAGE_BUCKET=...
FIREBASE_MESSAGING_SENDER_ID=...
FIREBASE_APP_ID=...
FIREBASE_MEASUREMENT_ID=...
```

### Translation Service (`.env`)
```
GROQ_API_KEY=...
```

### Sign Segmentation Service
- No required environment variables (uses defaults)

---

## Testing

### API Gateway
```bash
cd apiGateway
npm test
# Mocha looks for test files; uses chai, supertest, sinon
```

### Model
```bash
cd model
pytest test_main.py
```

### Sign Segmentation Service
```bash
cd signSegmentationService
pytest test_main.py   # Add tests as needed
```

---

## Common Gotchas & Debugging

1. **Frontend WebSocket Connection Fails**
   - Check `apiUrl` and `wsUrl` in `frontend/app.json`
   - WebSocket port is 9898, not 8080
   - Ensure backend is running: `npm run dev` in apiGateway

2. **API Gateway Build Fails**
   - Run `npm run build` explicitly; `tsc-alias` must run after `tsc`
   - Clear `dist/` if stale: `rm -rf dist && npm run build`

3. **Model Loading Issues**
   - `model.keras` (41MB) must exist; not committed to git
   - Check `.env` for missing paths
   - Verify MediaPipe/OpenCV are installed: `pip list | grep -E "mediapipe|opencv"`

4. **Service Connection Errors**
   - Docker Compose must be running: `docker-compose up -d`
   - Use service names as hostnames (e.g., `signsegmentationservice:8000`)
   - Check dependent services are healthy: `docker-compose ps`

5. **Python Service Dependency Conflicts**
   - Each service has its own `requirements.txt`
   - Use separate venvs: `.venv` per service or `--prefix` flag
   - TensorFlow + Keras are heavy; use Python 3.9–3.13 (see Dockerfiles)

6. **Frontend Camera Permissions**
   - Android: `app.json` already requests `android.permission.CAMERA`
   - iOS: Must accept runtime permission prompts
   - Emulator: May need to grant permissions in settings

7. **Sign Segmentation Tuning**
   - Adjust `low_factor`, `high_factor`, `still_frames_required` in `motion_detector.py`
   - Default values work for ~30fps video with moderate signing speed
   - Increase `still_frames_required` for slower signing or noisy backgrounds
   - Decrease for faster signing (but risk false positives)

---

## Architecture Notes

- **Microservices**: Each service has its own `package.json` or `requirements.txt`
- **Inter-Service Communication**: HTTP/REST for all service-to-service calls
- **File Routing**: Frontend uses Expo Router with files in `frontend/app/` (not config-based)
- **TypeScript Path Aliases**: API Gateway uses `@src/*` → relative paths (resolved at compile time)
- **ML Model Inference**: Served via separate Python service; not embedded in API Gateway
- **WebSocket**: Separate port (9898) for real-time client communication
- **State Management**: 
  - API Gateway holds session state (UUID → WebSocket, buffered signs)
  - Sign Segmentation Service holds per-frame motion state only (no session data)
  - Translation Service is stateless (accepts signs, returns translations)

---

## Deployment Notes

- **Docker Images**: All services have Dockerfiles; push to `baronocasiones/*` registry
- **API Endpoints**: Hardcoded in `frontend/app.json`; must be updated for production IP/domain
- **Redis**: Used for optional session persistence; can be removed for in-memory MVP
- **Model File**: Must be available at runtime in `model/model.keras`
- **Service Order**: Start translationService and signSegmentationService before API Gateway for best results
