# Gestura — Agent Instructions

Microservices + Expo mobile. Gesture recognition (ASL→English).

## Directory Layout

```
apiGateway/           # Express.js + TypeScript, two servers: REST (:8080) + WS (:9898)
frontend/             # Expo 54, React Native 0.81, expo-router (file-based)
translationService/   # FastAPI Python :7860 — segmentation + ML + grammar fix
model/                # Keras/TensorFlow 2.20 — BiLSTM+Transformer training pipeline
docs/                 # 5 markdown docs: API.md, ARCHITECTURE.md, AUTHENTICATION.md, DEPLOYMENT.md, ML_PIPELINE.md
scripts/              # Offline pipeline tools: segmentation_tester.py, test_pipeline.py (import from translationService/)
```

- `translationService/gestura/` — Legacy Groq-based version (HuggingFace Space, self-contained, not active service).
- `authenticationService/` — **Does not exist.** Auth lives in `apiGateway/routes/auth.ts` + `middleware/auth.ts`.

## Commands That Matter

### API Gateway
```bash
cd apiGateway
npm run dev        # nodemon + ts-node -r tsconfig-paths/register ./bin/www.ts
npm run build      # rm -rf dist && tsc && tsc-alias — required before `npm start`
npm test           # BUILD FIRST: npm run build && mocha --timeout 30000 dist/test/**/*.js
npm start          # node dist/bin/www.js (needs build first)
```

### Frontend
```bash
cd frontend
npx expo start     # dev menu — 'a' (Android), 'i' (iOS), 'w' (web)
npm run lint       # ESLint via expo
```

### Translation Service
```bash
cd translationService
uvicorn main:app --host 0.0.0.0 --port 7860 --reload
pytest test_main.py test_cache.py test_asl_dataset_generator.py -v
```

### Model
```bash
cd model
python model.py    # training
python ModelTester.py  # live webcam inference
# No tests exist — no test_*.py files in model/
```

## Architecture Gotchas

- **Two servers in apiGateway**: Express REST on `:8080`, WebSocket on `:9898`. WS server starts in `routes/api.ts`, **not** `bin/www.ts`.
- **`bin/www.ts` loads `.env`** via custom `loadEnvFile()` at startup (not `dotenv`).
- **`JWT_EXPIRES_IN` is hardcoded** in `routes/auth.ts` (7 days as `7 * 24 * 60 * 60`). The env var `JWT_EXPIRES_IN` in docker-compose is **never read** by the code.
- **`MONGODB_URI` is NOT in `.env`** — hardcoded fallback `'mongodb://localhost:27017'` in `routes/auth.ts:24`. Set via docker-compose env.
- **WS requires JWT**: Clients pass `?token=<jwt>&uuid=<uuid>` as query params on connect.
- **`POST /api/convert`** uses `multer` (`upload.single('rawImage')`), forwards JPEG as base64 to translation service `/process-frame`.
- **`GET /api/stop/:uuid`** needs an active WS session for that UUID in `sessionMap`, or returns 404.
- **No `GET /health` endpoint** in API Gateway (exists only in Translation Service).
- **Custom logging middleware**: `middleware/logging.ts` provides `requestLogger`, `responseLogger`, `withLogging(name)`, `setupWsLogging`. Winston logger in `services/logger.ts` with daily rotate. Replaces morgan entirely.
- **Unused deps in apiGateway**: `redis`, `morgan`, `jade` are installed but never imported.
- **Translation service has 5 endpoints**: `POST /process-frame`, `POST /stop`, `POST /translate`, `POST /convert-sentence`, `GET /health`.
- **Translation service sessions** hold per-uuid state: `MotionDetector` + `predicted_words` list.
- **Grammar fixer** uses **FLAN-T5** (`google/flan-t5-small` or local `models/flan-t5-asl-mini`), not Groq. Includes `asl_dataset_generator.py` and `finetune_asl_flan_t5.py` for fine-tuning. Has a `SemanticCache` with `sentence-transformers` (all-MiniLM-L6-v2).
- **MediaPipe model auto-downloads**: `holistic_landmarker.task` (~13MB) fetched from GCP on first run if missing.
- **Feature vector**: `lh(63) + rh(63) + pose(132) = 258 dims`. No face landmarks. Nose-normalized for translation invariance.
- **Frontend token storage**: In-memory only (`services/token.ts`). **NOT persisted** to AsyncStorage/SecureStore. Tokens lost on app restart.
- **Frontend hooks split**: `useGesturaAPI` (HTTP frame uploads via POST) + `useGesturaWebSocket` (receive-only WS for translations). Separate concerns.
- **Firebase config in frontend `.env`** but Firebase SDK is **never imported** anywhere. All auth is JWT-based against API Gateway.

## Test Quirks

- **Auth tests** (`auth.test.ts`) use `mongodb-memory-server` — **no external MongoDB needed**.
- **Integration tests** (`segmentation.integration.test.ts`) mock `axios` but **require WS server** on `:9898` (connect a real WebSocket). Run apiGateway dev server first.
- **Legacy test file** (`test.js` at apiGateway root) references Kafka and wrong routes — stale, will fail.
- **Translation service tests** (`test_main.py`, `test_cache.py`, `test_asl_dataset_generator.py`, `test_finetune_asl_flan_t5.py`) mock `keras.models.load_model` and `converter` module at import time (before `import main`).
- **No model tests** — `model/` has zero `test_*.py` files.

## TypeScript & Build

- **apiGateway**: `module: NodeNext`, `strict: true`, `noUncheckedIndexedAccess`, `exactOptionalPropertyTypes`. Path alias `@src/*` → `./*`. Build requires `tsc-alias` post-compilation.
- **Frontend**: `@/*` → `./*`. Extends `expo/tsconfig.base`.

## Python Version

- Local: 3.11.9 (`.python-version`)
- Docker: `python:3.12-slim`
- Translation Service Dockerfile skips `.venv` — `pip install` in empty container.
- Scripts at root (`scripts/`) require translationService/.venv: `source translationService/.venv/bin/activate && python scripts/segmentation_tester.py`.

## Frontend Config Quirks

- API URLs come from `config/environment.ts`, **not** `app.json.extra`. `DEV` mode uses hardcoded `192.168.100.5:8080`. `PROD` falls back to `app.json.extra`.
- Frontend has **EAS Build** setup (`eas.json`) with dev/preview/production profiles.
- Token storage is in-memory only (`services/token.ts`). **Not secure for production** — needs `expo-secure-store`.

## Docker Compose

```bash
docker compose up -d       # starts mongodb, translationService, apigateway
docker compose logs -f apigateway
docker compose down
```

- `apigateway` depends on `mongodb` + `translationService` but no health checks.
- `apigateway` uses a **prebuilt image** (`baronocasiones/gestura-apigateway:dev`), no `build` context in compose (Dockerfile is at `apiGateway/Dockerfile`).
- Translation Service Dockerfile: `CMD uvicorn main:app --host 0.0.0.0 --port 7860` — no `--reload`.
- Images: `baronocasiones/gestura-apigateway:dev`, `baronocasiones/gestura-translationservice:dev`
- **No CI/CD** — no `.github/workflows/`.

## Environment Variables

### API Gateway
| Var | Default | Note |
|-----|---------|------|
| `WEB_SOCKET_HOST` | `0.0.0.0` | |
| `WEB_SOCKET_PORT` | `9898` | |
| `TRANSLATION_SERVICE_URL` | `http://translationService:7860` | Docker service name |
| `JWT_SECRET` | `default-dev-secret-change-in-production` | Hardcoded fallback in code, not in `.env` |
| `MONGODB_URI` | `mongodb://localhost:27017` | Not in `.env`; hardcoded fallback in `routes/auth.ts` |
| `PORT` | `'8080'` | HTTP server port, read from env in `bin/www.ts` |

`JWT_EXPIRES_IN` (env var in docker-compose) is **never read** — expiration is hardcoded as 7 days.

### Translation Service
| Var | Default | Note |
|-----|---------|------|
| `FEATURE_DIM` | `258` | lh(63)+rh(63)+pose(132) |
| `WINDOW_SIZE` | `35` | Model input window |
| `PERSIST_WINDOW` | `5` | Bounded persistence frames |
| `IDLE_THRESHOLD` | `15` | Must be > `still_frames_required` (8) |
| `MD_STILLNESS_FLOOR` | `0.5` | Motion detector stillness floor |
| `FLAN_T5_MODEL` | `google/flan-t5-small` | Or local `models/flan-t5-asl-mini` |
| `CACHE_SIMILARITY_THRESHOLD` | `0.85` | Semantic cache cosine sim |

`GROQ_API_KEY` is **not used** by current code (legacy from `gestura/` subdirectory).

## Key Gotchas

- **Model file sizes**: `best_model.keras` is ~21MB (not 41MB), `model.keras` is ~41MB. Translation service loads `best_model.keras`.
- **Model files in both `model/` and `translationService/`** — must be manually synced or restored from checkpoint.
- **`.gitignore`** blocks `*.keras`, `translationService/models/`, `**/keypoint_data_*/`, `**/.env`, `**/Dockerfile`.
- **CUDA/XLA**: Scripts search for `libdevice.10.bc` before importing TF, sets `XLA_FLAGS`.
- **Bounded persistence**: `converter.py` fills last-known hand positions for up to `PERSIST_WINDOW` frames, then decays to zeros.
- **Motion detector tuning**: `still_frames_required=8`, `motion_smoothing=0.6`, `stillness_floor=0.5`, adaptive thresholds via median of last 30 frames.
- **`normalize.py`**: Variable-length sequences → 35 frames. Downsampling with round-robin indices, upsampling by padding last frame.
- **Root `package.json`** only has `axios` (likely orphaned — each service has its own `package.json`).
