# Gestura

[![Node.js](https://img.shields.io/badge/Node.js-18+-3c873a?style=flat-square&logo=node.js&logoColor=white)](https://nodejs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.9-blue?style=flat-square&logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![Expo](https://img.shields.io/badge/Expo-54-000?style=flat-square&logo=expo&logoColor=white)](https://expo.dev)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.120-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

**Gestura** is a real-time sign language recognition platform that translates American Sign Language (ASL) gestures into natural English text. It combines a mobile camera app with a microservices backend powered by deep learning and LLM-based grammar correction.

[Architecture](#architecture) • [Services](#services) • [Quick Start](#quick-start) • [Development](#development) • [Tech Stack](#tech-stack) • [Troubleshooting](#troubleshooting)

---

## Architecture

Gestura follows a microservices architecture with an Expo React Native mobile frontend and backend services orchestrated via Docker Compose.

```
┌──────────────┐     WebSocket/REST     ┌──────────────────┐
│  Mobile App  │ ──────────────────────▶│   API Gateway    │
│ Expo/RN      │◀──────────────────────│  Express.js      │
└──────────────┘                       │  :8080 / :9898   │
                                        └──────┬───────────┘
                                               │ HTTP
                                        ┌──────▼───────────┐
                                        │ Translation Svc  │──▶ Groq LLM
                                        │  FastAPI :7860   │──▶ Keras Model
                                        └──────────────────┘
                                               │
                                        ┌──────▼───────────┐
                                        │     MongoDB      │
                                        │     :27017       │
                                        └──────────────────┘
```

**Data flow:**

1. The mobile app captures video frames via the device camera
2. Frames are sent to the API Gateway over WebSocket, then forwarded to the Translation Service
3. The Translation Service extracts keypoints using MediaPipe Holistic, detects sign boundaries via motion analysis, and runs ML inference on each segmented sign
4. Predicted words are buffered per session; when the user stops signing, the Groq LLM converts ASL gloss into natural English
5. Translated text is returned to the mobile app via WebSocket

---

## Services

| Service | Language | Framework | Port | Description |
|---------|----------|-----------|------|-------------|
| **Frontend** | TypeScript | Expo / React Native 0.81 | - | Mobile camera app for sign capture and display |
| **API Gateway** | TypeScript | Express.js | `8080` (REST), `9898` (WS) | Routes requests, manages WebSocket connections |
| **Translation Service** | Python | FastAPI | `7860` | Sign segmentation, ML inference, Groq grammar correction |
| **ML Model** | Python | Keras / TensorFlow 2.20 | - | LSTM/Transformer model for gesture classification |
| **MongoDB** | - | MongoDB 7.0 | `27017` | User data and session storage |

### Frontend

An Expo mobile application that captures live video using `react-native-vision-camera`, processes frames via `react-native-fast-opencv`, and streams them to the backend for real-time sign language recognition. Built with Expo Router for file-based navigation.

### API Gateway

Express.js server acting as the entry point for all client connections. Handles REST API requests on port `8080` and manages persistent WebSocket connections on port `9898`. Routes translation requests to the Translation Service.

### Translation Service

FastAPI application combining three processing stages:

- **Sign Segmentation** — Uses MediaPipe Holistic to extract 258-dimensional keypoint features (hands + pose) per frame, with motion boundary detection to isolate individual signs
- **ML Inference** — Runs a Keras LSTM/Transformer model on each segmented sign window (35 frames) to predict the ASL word
- **Grammar Correction** — Sends accumulated ASL gloss to Groq's llama-3.3-70b model to produce natural English sentences

### ML Model

A hybrid LSTM-Transformer architecture trained on ASL keypoint data. The pipeline includes data augmentation, nose-normalization for view invariance, and variable-length sequence handling.

---

## Quick Start

### Prerequisites

- [Node.js](https://nodejs.org) 18+
- [Python](https://python.org) 3.12
- [Docker](https://docker.com) and Docker Compose
- [Expo CLI](https://docs.expo.dev/more/create-expo-app/) (`npx expo`)
- A [Groq API key](https://console.groq.com) for grammar correction

### 1. Start the Backend Services

```bash
# Clone the repository
git clone https://github.com/your-org/gestura.git
cd gestura

# Set up environment variables
cp translationService/.env.example translationService/.env
# Edit translationService/.env and add your GROQ_API_KEY

# Start all services with Docker Compose
docker compose up -d

# Verify services are running
docker compose ps
```

### 2. Start the Frontend

```bash
cd frontend
npm install
npx expo start
```

Press `a` for Android emulator, `i` for iOS simulator, or `w` for web.

> [!TIP]
> Update the API endpoints in `frontend/app.json` under `extra.apiUrl` and `extra.wsUrl` to match your backend's IP address before connecting from a physical device.

---

## Development

### API Gateway

```bash
cd apiGateway
npm install
npm run dev          # Development with hot reload (port 8080, WS 9898)
npm run build        # TypeScript compilation
npm test             # Run Mocha tests
```

### Translation Service

```bash
cd translationService
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

### ML Model

```bash
cd model
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py       # Run inference or training
pytest test_main.py  # Run tests
```

### Frontend

```bash
cd frontend
npm install
npx expo start       # Dev server
npm run lint         # ESLint
npm run android      # Build for Android
```

---

## Environment Variables

| Service | Variable | Default | Description |
|---------|----------|---------|-------------|
| **API Gateway** | `WEB_SOCKET_HOST` | `0.0.0.0` | WebSocket bind address |
| | `WEB_SOCKET_PORT` | `9898` | WebSocket server port |
| | `TRANSLATION_SERVICE_URL` | `http://translationService:7860` | Translation Service endpoint |
| **Translation Service** | `GROQ_API_KEY` | — | API key for Groq LLM |
| | `FEATURE_DIM` | `258` | Keypoint feature dimension |
| | `WINDOW_SIZE` | `35` | Model input window size |
| | `PERSIST_WINDOW` | `5` | Bounded persistence frames |
| **Frontend** | `FIREBASE_API_KEY` | — | Firebase auth config (in `.env`) |
| | `FIREBASE_AUTH_DOMAIN` | — | |
| | `FIREBASE_PROJECT_ID` | — | |
| | `FIREBASE_STORAGE_BUCKET` | — | |
| | `FIREBASE_MESSAGING_SENDER_ID` | — | |
| | `FIREBASE_APP_ID` | — | |
| | `FIREBASE_MEASUREMENT_ID` | — | |

---

## Docker Compose

```bash
# Start all services
docker compose up -d

# View logs for a specific service
docker compose logs -f apigateway

# Stop all services
docker compose down
```

**Docker images** (for CI/CD):
- `baronocasiones/gestura-apigateway:dev`
- `baronocasiones/gestura-translationservice:dev`

---

## Testing

```bash
# API Gateway tests (Mocha + Chai + Sinon)
cd apiGateway && npm test

# Translation Service tests (pytest)
cd translationService && pytest test_main.py -v

# ML Model tests
cd model && pytest test_main.py
```

---

## Project Structure

```
gestura/
├── apiGateway/              # Express.js API Gateway (TypeScript)
│   ├── bin/www.ts           # Server entry point
│   ├── app.ts               # Express app setup
│   ├── routes/              # API endpoints
│   ├── test/                # Mocha tests
│   └── Dockerfile
├── frontend/                # Expo React Native app
│   ├── app/                 # File-based routes (Expo Router)
│   ├── assets/              # Images, icons, fonts
│   └── app.json             # Expo configuration
├── translationService/      # Python FastAPI service
│   ├── main.py              # FastAPI server + session management
│   ├── converter.py         # MediaPipe keypoint extraction
│   ├── motion_detector.py   # Sign boundary detection
│   ├── normalize.py         # Frame normalization
│   ├── model.keras          # Trained model weights
│   └── Dockerfile
├── model/                   # ML model training pipeline
│   ├── model.py             # LSTM/Transformer architecture
│   ├── DataAugmentor.py     # Data augmentation
│   ├── DataNormalizer.py    # Keypoint normalization
│   └── keypoint_data*/      # Training datasets
├── docker-compose.yml       # Service orchestration
└── AGENTS.md                # AI agent setup guide
```

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| **Mobile** | Expo 54, React Native 0.81, Expo Router |
| **Camera & Vision** | react-native-vision-camera, react-native-fast-opencv |
| **Backend (Node.js)** | Express.js, TypeScript, ws (WebSocket) |
| **Backend (Python)** | FastAPI, Uvicorn, Keras/TensorFlow 2.20 |
| **ML & CV** | MediaPipe Holistic, LSTM-Transformer, OpenCV |
| **LLM** | Groq API (llama-3.3-70b) |
| **Database** | MongoDB 7.0 |
| **Infrastructure** | Docker Compose |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **WebSocket connection fails** | Verify `wsUrl` in `frontend/app.json` matches the server IP. WebSocket port is `9898`, not `8080`. |
| **API Gateway build fails** | Run `rm -rf dist && npm run build`. The `tsc-alias` step is required after compilation. |
| **Model not found** | Ensure `model.keras` or `best_model.keras` exists in `translationService/`. Large files (~41 MB) not committed to git. |
| **Docker services can't connect** | Use service names as hostnames (`translationService:7860`, `mongodb:27017`). Check `docker compose ps`. |
| **Camera permission denied** | Android: permissions in `app.json`. iOS: accept runtime prompts. Emulator: check settings. |
| **Groq API errors** | Verify `GROQ_API_KEY` is set in `translationService/.env`. |
| **Python dependency conflicts** | Use separate virtual environments per service. TensorFlow requires Python 3.12. |
