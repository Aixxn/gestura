# Agent Instructions for Gestura

## Project Structure: Microservices + Expo Mobile

**Gestura** is a gesture recognition platform with:
- **Frontend**: Expo React Native app (camera capture, sign language recognition)
- **Backend Services**: TypeScript/Python microservices + Python ML pipeline
- **Services**: API Gateway, Authentication Service, Translation Service (merged with sign segmentation), ML Model
- **Infrastructure**: Docker Compose

### Directory Layout

```
apiGateway/          → Express.js API gateway (Node.js, TypeScript)
frontend/            → Expo app with file-based routing (React Native)
translationService/  → FastAPI service for gesture translation (segmentation + ML + Groq) (Python)
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
- `middleware/auth.ts` → JWT authentication middleware
- `types/index.ts` → Shared TypeScript interfaces
- `tsconfig.json` → Includes `baseUrl: "."`, paths: `@src/*` → `./`

**Environment** (`.env`):
```
WEB_SOCKET_HOST = "0.0.0.0"
WEB_SOCKET_PORT = 9898
TRANSLATION_SERVICE_URL = "http://translationService:7860"
JWT_SECRET = "your-secret-key-change-in-production"
JWT_EXPIRES_IN = "7d"
```

**Gotchas**:
- Uses `tsc-alias` for path resolution post-compilation (required for `@src/*` imports)
- Strict TypeScript (`strict: true`, `noUnusedLocals`, etc.)
- WebSocket server on separate port 9898 from API (8080)

---

## Authentication (JWT)

**Tech**: bcrypt (password hashing), jsonwebtoken (JWT), MongoDB (user storage)

### Auth Flow
1. **Register**: `POST /api/auth/register` with `{ email, password, full_name? }` → returns `{ token, user }`
2. **Login**: `POST /api/auth/login` with `{ email, password }` → returns `{ token, user }`
3. **Authenticated requests**: Include header `Authorization: Bearer <token>`
4. **Get profile**: `GET /api/auth/me` (requires token) → returns current user
5. **Logout**: `POST /api/auth/logout` (stateless no-op on server)

### Protected Routes
- `POST /api/convert` — requires valid JWT
- `GET /api/stop/:uuid` — requires valid JWT
- `GET /api/auth/me` — requires valid JWT
- WebSocket `ws://...:9898` — requires `token` query param

Public routes:
- `POST /api/auth/register`
- `POST /api/auth/login`
- `POST /api/auth/logout`
- `GET /health`

### Key Components
- `middleware/auth.ts` → `requireAuth` middleware that verifies `Authorization: Bearer <token>`
- `routes/auth.ts` → bcrypt hashing (12 rounds), JWT signing with 7-day expiry, email-based login
- `types/index.ts` → `IUser`, `JwtPayload`, `AuthRequest` interfaces

### Testing
- Auth tests require MongoDB running (local or Docker)
- Segmentation tests mock axios and don't need MongoDB
- Tests use `makeToken()` helper to generate valid JWTs for protected route tests

### Environment Variables
```
JWT_SECRET=your-secret-key           # Required - used to sign/verify tokens
JWT_EXPIRES_IN=7d                    # Optional - token expiry duration
```

---

## Translation Service (FastAPI + Python)

**Tech**: FastAPI, Keras/TensorFlow, MediaPipe Holistic, Groq LLM API (for ASL grammar correction)

### Development
```bash
cd translationService
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

**Key Functionality**:
- Sign segmentation via MediaPipe Holistic (keypoint extraction + motion boundary detection)
- Real-time ML inference: when a sign ends, immediately predicts the word
- Buffers predicted words per session
- Uses Groq LLM API (llama-3.3-70b) to fix grammar on session end
- FastAPI debug mode enabled in `main.py`

**Key Components**:
- `main.py` → FastAPI server with session management
- `converter.py` → MediaPipe keypoint extraction, nose-normalization, **bounded persistence** (last-known hand positions fill in up to `PERSIST_WINDOW` frames, then decay to zeros)
- `motion_detector.py` → Hysteresis-based sign boundary detection
- `normalize.py` → Variable-length sign normalization to model's window size

**Endpoints**:
- `POST /process-frame` → Receive frame, run segmentation + ML inference
- `POST /stop` → Finalize session, run Groq grammar correction
- `POST /translate` → Standalone ML inference on pre-normalized window
- `POST /convert-sentence` → Standalone Groq grammar fix
- `GET /health` → Health check

**Environment** (`.env`):
- `GROQ_API_KEY` required for LLM integration
- `FEATURE_DIM=258` (hands + pose, no face)
- `WINDOW_SIZE=35` (model expects 35-frame input windows)
- `PERSIST_WINDOW=5` (bounded persistence — frames before undetected hand decays to zeros)

**Port**: 7860 (FastAPI default)

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
- `mongodb` → MongoDB for auth

### Local Development
```bash
docker compose up -d
docker compose logs -f apigateway
docker compose down
```

**Image Names** (for CI/CD):
- `baronocasiones/gestura-apigateway:dev`
- `baronocasiones/gestura-translationservice:dev`

---

## Common Commands by Service

### API Gateway
```bash
cd apiGateway
npm run dev         # Development (watch mode)
npm run build       # TypeScript → dist/
npm test            # Mocha tests (requires MongoDB running)
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
TRANSLATION_SERVICE_URL=http://translationService:7860
JWT_SECRET=your-secret-key-change-in-production
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
FEATURE_DIM=258
WINDOW_SIZE=35
```

---

## Testing

### API Gateway
```bash
cd apiGateway
npm test
# Mocha looks for test files; uses chai, supertest, sinon
# Requires MongoDB running for auth tests
# Segmentation tests use mocked axios and don't need MongoDB
```

### Model
```bash
cd model
pytest test_main.py
```

### Translation Service
```bash
cd translationService
pytest test_main.py -v
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
   - Docker Compose must be running: `docker compose up -d`
   - Use service names as hostnames (e.g., `translationService:7860`)
   - Check dependent services are healthy: `docker compose ps`

5. **Python Service Dependency Conflicts**
   - Translation Service has its own `requirements.txt`
   - Use separate venv: `.venv` per service or `--prefix` flag
   - TensorFlow + Keras are heavy; use Python 3.12 (see Dockerfile)

6. **Frontend Camera Permissions**
   - Android: `app.json` already requests `android.permission.CAMERA`
   - iOS: Must accept runtime permission prompts
   - Emulator: May need to grant permissions in settings

7. **Sign Segmentation Tuning**
   - Adjust `low_factor`, `high_factor`, `still_frames_required` in `translationService/motion_detector.py`
   - Default values work for ~30fps video with moderate signing speed
   - Increase `still_frames_required` for slower signing or noisy backgrounds
   - Decrease for faster signing (but risk false positives)

8. **JWT Authentication Issues**
   - `JWT_SECRET` must be set in environment — default is insecure
   - Auth tests require MongoDB running (start with `docker compose up -d mongodb`)
   - Frontend stores token in memory only by default; use `expo-secure-store` for persistence
   - Token expiry defaults to 7 days; configure via `JWT_EXPIRES_IN` env var
   - WebSocket connections require `token` query param alongside `uuid`

---

## Architecture Notes

- **Microservices**: Each service has its own `package.json` or `requirements.txt`
- **Inter-Service Communication**: HTTP/REST for all service-to-service calls
- **File Routing**: Frontend uses Expo Router with files in `frontend/app/` (not config-based)
- **TypeScript Path Aliases**: API Gateway uses `@src/*` → relative paths (resolved at compile time)
- **ML Model Inference**: Served via Translation Service; segmentation + ML + Groq in one pipeline
- **WebSocket**: Separate port (9898) for real-time client communication
- **State Management**: 
  - API Gateway holds session WebSocket connections only (no sign buffering)
  - Translation Service holds all per-session state (motion detector, accumulated predicted words)
- **Authentication**: JWT-based, server-side only (API Gateway + MongoDB). Password hashing via bcrypt (12 rounds). All translation endpoints require valid JWT.

---

## Deployment Notes

- **Docker Images**: All services have Dockerfiles; push to `baronocasiones/*` registry
- **API Endpoints**: Hardcoded in `frontend/app.json`; must be updated for production IP/domain
- **Model File**: Must be available at runtime in `translationService/model.keras`
- **Service Order**: Start translationService before API Gateway for best results
