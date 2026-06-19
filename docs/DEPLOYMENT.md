# Gestura Deployment Guide

## Architecture Overview

```
┌──────────────────────────┐
│    Docker Host           │
│                          │
│  ┌───────────────────┐   │
│  │   nginx (opt.)    │   │  Port 80/443 → 8080
│  └──────┬────────────┘   │
│         │                 │
│  ┌──────▼────────────┐   │
│  │   API Gateway      │   │  Port 8080 (HTTP), :9898 (WS)
│  │   (Express/TS)     │   │
│  └──────┬────────────┘   │
│         │                 │
│  ┌──────▼────────────┐   │
│  │ Translation Svc    │   │  Port 7860 (internal)
│  │   (FastAPI/Python) │   │
│  └───────────────────┘   │
│                          │
│  ┌───────────────────┐   │
│  │    MongoDB 7.0     │   │  Port 27017 (internal)
│  └───────────────────┘   │
└──────────────────────────┘
```

---

## Prerequisites

- Docker Engine 24+ and Docker Compose v2+
- Node.js 24+ (for local API Gateway dev)
- Python 3.12+ (for local Translation Service dev)
- Expo CLI (for frontend dev)

---

## Docker Compose Deployment

### Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd gestura

# Start all services
docker compose up -d

# Check logs
docker compose logs -f
```

### Service Configuration

The `docker-compose.yml` defines three services:

#### MongoDB
```yaml
mongodb:
  image: mongo:7.0
  ports:
    - "27017:27017"    # Exposed for local dev; remove in production
  volumes:
    - mongodb_data:/data/db   # Persistent storage
```

#### API Gateway
```yaml
apigateway:
  image: baronocasiones/gestura-apigateway:dev
  ports:
    - "8080:8080"       # HTTP API
    - "9898:9898"       # WebSocket
  environment:
    - MONGODB_URI=mongodb://mongodb:27017
    - TRANSLATION_SERVICE_URL=http://translationService:7860
    - JWT_SECRET=gestura-dev-jwt-secret-change-in-production
```

#### Translation Service
```yaml
translationService:
  image: baronocasiones/gestura-translationservice:dev
  ports:
    - "7860:7860"       # Internal API (should not be exposed publicly)
```

---

## Environment Variables

### API Gateway (`.env`)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `WEB_SOCKET_HOST` | No | `0.0.0.0` | WebSocket bind address |
| `WEB_SOCKET_PORT` | No | `9898` | WebSocket listen port |
| `TRANSLATION_SERVICE_URL` | Yes | `http://translationService:7860` | Internal translation service URL |
| `JWT_SECRET` | **Yes** | `default-dev-secret-...` | JWT signing secret — **change in production** |
| `JWT_EXPIRES_IN` | No | `7d` | Token expiry duration |
| `MONGODB_URI` | No | `mongodb://localhost:27017` | MongoDB connection string |
| `PORT` | No | `8080` | HTTP listen port |

### Translation Service (`.env`)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | No | — | Groq API key (not currently used; uses local FLAN-T5) |
| `FEATURE_DIM` | No | `258` | Keypoint feature dimension |
| `WINDOW_SIZE` | No | `35` | ML model window size |
| `PERSIST_WINDOW` | No | `5` | Bounded persistence frame count |
| `IDLE_THRESHOLD` | No | `15` | Frames before idle detection |
| `MD_STILLNESS_FLOOR` | No | `0.5` | Motion detector stillness floor |
| `FLAN_T5_MODEL` | No | `google/flan-t5-small` | Grammar correction model |

### Frontend (`app.json` extra / environment.ts)

| Variable | Where | Description |
|----------|-------|-------------|
| `apiUrl` | `app.json` extra | Backend HTTP URL (e.g., `http://192.168.1.100:8080`) |
| `wsUrl` | `app.json` extra | Backend WebSocket URL (e.g., `ws://192.168.1.100:9898`) |

The frontend uses `config/environment.ts` which reads from `app.json` extra in production, or hardcoded dev IPs in development (`__DEV__` flag).

---

## Building Docker Images

### API Gateway
```bash
cd apiGateway
docker build -t baronocasiones/gestura-apigateway:dev .
docker push baronocasiones/gestura-apigateway:dev  # for remote deployment
```

### Translation Service
```bash
cd translationService
docker build -t baronocasiones/gestura-translationservice:dev .
docker push baronocasiones/gestura-translationservice:dev
```

---

## Production Deployment Checklist

### 1. Security
- [ ] Change `JWT_SECRET` to a strong random value
- [ ] Use `wss://` for WebSocket (requires TLS termination via nginx/caddy)
- [ ] Use `https://` for all API endpoints
- [ ] Do NOT expose port 7860 (Translation Service) to the internet
- [ ] Do NOT expose port 27017 (MongoDB) to the internet
- [ ] Add rate limiting to auth endpoints
- [ ] Add MongoDB authentication (username/password)
- [ ] Use `expo-secure-store` on frontend instead of in-memory token storage

### 2. Networking

Expose only port 8080 (and optionally 9898) through a reverse proxy:

```
Internet → nginx (443/80) → API Gateway (8080)
                        → WebSocket (9898) [or proxy via nginx]
```

**nginx config snippet:**
```nginx
server {
    listen 443 ssl;
    server_name api.gestura.example.com;

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /ws {
        proxy_pass http://localhost:9898;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### 3. Frontend Configuration

Update the API URLs in `frontend/app.json`:
```json
{
  "extra": {
    "apiUrl": "https://api.gestura.example.com",
    "wsUrl": "wss://api.gestura.example.com/ws"
  }
}
```

### 4. Data Persistence

MongoDB data persists via Docker volumes (`mongodb_data`). To back up:
```bash
docker exec gestura-mongodb-1 mongodump --out /tmp/backup
docker cp gestura-mongodb-1:/tmp/backup ./mongodb-backup-$(date +%Y%m%d)
```

### 5. Model Files

Ensure these files exist in `translationService/`:
- `best_model.keras` (or `model.keras`) — ML model weights
- `sign_classes.npy` — Class label mapping
- `holistic_landmarker.task` — MediaPipe model (auto-downloaded if missing)

These are large files (~21 MB + ~13 MB) and should be included in the Docker image or mounted as a volume.

---

## Local Development

### API Gateway
```bash
cd apiGateway
npm install
npm run dev       # nodemon + ts-node (port 8080 + WebSocket 9898)
```

### Translation Service
```bash
cd translationService
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

### Full Stack with Docker
```bash
docker compose up -d
# API Gateway: http://localhost:8080
# WebSocket: ws://localhost:9898
# Translation Service: http://localhost:7860 (internal)
```

### Frontend
```bash
cd frontend
npm install
npx expo start
# Press 'a' for Android emulator, 'i' for iOS simulator, 'w' for web
```

---

## Frontend Builds

### Android
```bash
cd frontend
npx expo run:android   # Requires Android Studio + SDK
```

### iOS (macOS only)
```bash
cd frontend
npx expo run:ios       # Requires Xcode
```

### OTA Updates (EAS)
The project is configured for Expo updates:
```bash
npx eas update --branch production --message "Deploy v1.2"
```

---

## Troubleshooting

### Service Connection Issues
```
# Check if services are running
docker compose ps

# View logs
docker compose logs -f apigateway
docker compose logs -f translationService

# Test internal connectivity
docker compose exec apigateway curl http://translationService:7860/health
```

### WebSocket Issues
- Verify `wsUrl` in frontend config matches the server IP
- WebSocket port is 9898, NOT 8080
- Ensure JWT token is valid and included as `?token=` query param
- Check for firewalls blocking port 9898

### Model Loading Failures
```
# Check model files exist
docker compose exec translationService ls -la /translationService/best_model.keras

# Verify model compatibility
docker compose exec translationService python -c "
import keras; m = keras.models.load_model('/translationService/best_model.keras')
print(m.summary())
"
```

### MongoDB Connection Issues
```
# Verify MongoDB is running
docker compose ps mongodb

# Check connection from API Gateway
docker compose exec apigateway node -e "
const { MongoClient } = require('mongodb');
new MongoClient('mongodb://mongodb:27017').connect()
  .then(c => { console.log('Connected'); c.close(); })
  .catch(e => console.error(e));
"
```

### CUDA/GPU Issues (Translation Service)
The Translation Service runs on CPU by default (`CUDA_VISIBLE_DEVICES=-1`). For GPU acceleration:
```yaml
# docker-compose.yml
translationService:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

---

## Service Startup Order

1. **MongoDB** starts first (no dependencies)
2. **Translation Service** starts (no dependencies, but takes ~30s to load models)
3. **API Gateway** starts last (depends on both MongoDB and Translation Service)

The API Gateway will crash-loop until MongoDB is ready. This is expected — Docker will restart it automatically (`restart: always`).
