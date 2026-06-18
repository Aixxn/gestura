# Gestura Authentication

## Overview

Gestura uses **JWT-based authentication** with bcrypt password hashing. The API Gateway handles all auth logic with MongoDB as the user store. Tokens are stored **in-memory** on the frontend (with expo-secure-store recommended for production).

---

## Auth Flow

### Registration
```
Frontend                          API Gateway                     MongoDB
   │                                  │                              │
   │  POST /api/auth/register         │                              │
   │  { email, password, full_name }  │                              │
   │ ──────────────────────────────►  │                              │
   │                                  │  Check existing email        │
   │                                  │ ────────────────────────────► │
   │                                  │ ◄──────────────────────────── │
   │                                  │                              │
   │                                  │  bcrypt.hash(password, 12)   │
   │                                  │                              │
   │                                  │  INSERT user                 │
   │                                  │ ────────────────────────────► │
   │                                  │ ◄──────────────────────────── │
   │                                  │                              │
   │                                  │  jwt.sign({ userId, email }, │
   │                                  │    JWT_SECRET, 7d)           │
   │                                  │                              │
   │  201 { token, user }             │                              │
   │ ◄────────────────────────────── │                              │
   │                                  │                              │
   │  Store token in memory           │                              │
```

### Login
```
Frontend                          API Gateway                     MongoDB
   │                                  │                              │
   │  POST /api/auth/login            │                              │
   │  { email, password }             │                              │
   │ ──────────────────────────────►  │                              │
   │                                  │  FIND user by email          │
   │                                  │ ────────────────────────────► │
   │                                  │ ◄──────────────────────────── │
   │                                  │                              │
   │                                  │  bcrypt.compare(password,    │
   │                                  │    stored_hash)              │
   │                                  │                              │
   │                                  │  jwt.sign({ userId, email }, │
   │                                  │    JWT_SECRET, 7d)           │
   │                                  │                              │
   │  200 { token, user }             │                              │
   │ ◄────────────────────────────── │                              │
   │                                  │                              │
   │  Store token in memory           │                              │
```

### Authenticated Request
```
Frontend                          API Gateway
   │                                  │
   │  GET /api/auth/me                │
   │  Authorization: Bearer <token>   │
   │ ──────────────────────────────►  │
   │                                  │  jwt.verify(token, JWT_SECRET)
   │                                  │  ┌─ Success → attach user to req
   │                                  │  └─ Failure → 401
   │                                  │
   │  200 { user }                    │
   │ ◄────────────────────────────── │
```

### WebSocket Authentication
```
Frontend                          API Gateway (WS :9898)
   │                                  │
   │  ws://host:9898?uuid=...         │
   │       &token=<jwt>              │
   │ ──────────────────────────────►  │
   │                                  │  jwt.verify(token, JWT_SECRET)
   │                                  │  ┌─ Success → accept connection
   │                                  │  └─ Failure → ws.close(1008)
   │                                  │
   │  Connection established          │
   │ ◄────────────────────────────── │
```

---

## Token Details

| Property | Value |
|----------|-------|
| Algorithm | HS256 |
| Payload | `{ userId: string, email: string }` |
| Expiry | 7 days (configurable via `JWT_EXPIRES_IN`) |
| Storage | Frontend in-memory (see below) |

### JWT Payload
```json
{
  "userId": "664f1a2b3c4d5e6f7a8b9c0d",
  "email": "user@example.com",
  "iat": 1704067200,
  "exp": 1704672000
}
```

---

## Endpoint Protection Matrix

| Endpoint | Auth Required | Notes |
|----------|---------------|-------|
| `POST /api/auth/register` | No | |
| `POST /api/auth/login` | No | |
| `POST /api/auth/logout` | No | Stateless no-op |
| `GET /api/auth/me` | **Yes** | Returns current user profile |
| `POST /api/convert` | **Yes** | Frame upload endpoint |
| `GET /api/stop/:uuid` | **Yes** | Session finalization |
| `GET /health` | No | Public health check |
| WebSocket `ws://...` | **Yes** | Token via `?token=` query param |

---

## Frontend Token Management

### Current Implementation (In-Memory)

The frontend stores the JWT in a module-level variable (`services/token.ts`):

```typescript
// services/token.ts
let accessToken: string | null = null;

export function setToken(token: string | null) { accessToken = token; }
export function getToken(): string | null { return accessToken; }
export function clearToken() { accessToken = null; }
```

**Important:** This means the token is lost on app restart or page refresh. For production, replace with `expo-secure-store`:

```typescript
import * as SecureStore from 'expo-secure-store';

export async function setToken(token: string) {
  await SecureStore.setItemAsync('auth_token', token);
}
export async function getToken(): Promise<string | null> {
  return await SecureStore.getItemAsync('auth_token');
}
```

### Auth Header Attachment

The API client (`services/api.ts`) uses an axios interceptor to attach the token:

```typescript
apiClient.interceptors.request.use((config) => {
  const token = getToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});
```

The `/api/convert` endpoint uses raw `fetch()` instead of axios (for FormData uploads), but also manually attaches the Bearer header.

### Session Management

| Action | Token Handling |
|--------|---------------|
| Register | `setToken()` on success → redirect to camera screen |
| Login | `setToken()` on success → redirect to camera screen |
| Logout | `clearToken()` → redirect to login screen |
| 401 Response | Warning logged; user must re-authenticate |

---

## Security Considerations

1. **JWT_SECRET must be changed in production** — the default `'default-dev-secret-change-in-production'` is insecure.

2. **Password hashing** uses bcrypt with 12 salt rounds — computationally expensive but provides strong protection against brute force.

3. **Token is sent as query parameter** for WebSocket connections. Use `wss://` in production to encrypt the connection.

4. **No refresh token mechanism** — tokens expire after 7 days. The user must log in again. This could be enhanced with refresh tokens for production.

5. **Rate limiting** is not currently implemented. Consider adding rate limiting to auth endpoints for production deployment.

6. **Token storage** is in-memory only. Add `expo-secure-store` (iOS) / `EncryptedSharedPreferences` (Android) for persistent storage.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JWT_SECRET` | `default-dev-secret-...` | Secret key for signing JWTs |
| `JWT_EXPIRES_IN` | `7d` | Token expiry duration |

**Password complexity is enforced on the frontend** (backend only checks for non-empty values). Add server-side validation for production.

---

## MongoDB User Schema

Stored in the `gestura` database, `users` collection:

```json
{
  "_id": ObjectId,
  "email": "user@example.com",
  "password": "$2b$12$...bcrypt_hash...",
  "full_name": "Jane Doe",
  "created_at": ISODate
}
```

The `password` field is excluded from `/api/auth/me` responses via MongoDB projection `{ password: 0 }`.
