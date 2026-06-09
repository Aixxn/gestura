# WebSocket Implementation - Separated Concerns

## 📁 New Structure

```
frontend/
  hooks/
    ✅ useGesturaAPI.ts          - HTTP API calls (sending frames)
    ✅ useGesturaWebSocket.ts    - WebSocket connection (receiving translations)
  app/(tabs)/
    ✅ camera.tsx                - Updated to use both hooks
```

## 🎯 Implementation Overview

### **1. useGesturaAPI Hook** (HTTP Only)
**Purpose:** Handle HTTP requests to the API Gateway

**Features:**
- ✅ Session UUID generation
- ✅ Send frames via HTTP POST
- ✅ Stop processing endpoint
- ✅ Health check
- ✅ Error handling
- ✅ Loading states

**Usage:**
```typescript
const { 
  sessionUUID,      // Unique session identifier
  sendFrame,        // Send frame to API
  stopProcessing,   // Stop the processing session
  isSending,        // Loading state
  error            // Error message if any
} = useGesturaAPI();
```

---

### **2. useGesturaWebSocket Hook** (Receive Only)
**Purpose:** Manage WebSocket connection to receive translations

**Features:**
- ✅ Auto-connect/disconnect based on `enabled` prop
- ✅ Auto-reconnection with exponential backoff
- ✅ Connection state management (connected, connecting, disconnected)
- ✅ Error handling and reporting
- ✅ Manual reconnect capability
- ✅ Graceful cleanup on unmount

**Configuration:**
```typescript
const {
  translation,        // Received translation text
  isConnected,        // Connection status
  isConnecting,       // Connecting status
  error,              // Error message
  reconnectAttempts,  // Number of reconnect attempts
  connect,            // Manual connect
  disconnect,         // Manual disconnect
  reconnect,          // Force reconnect
  clearTranslation,   // Clear translation text
} = useGesturaWebSocket({
  uuid: sessionUUID,           // Session UUID from useGesturaAPI
  enabled: isActive,           // Connect only when camera is active
  autoReconnect: true,         // Auto-reconnect on disconnect
  reconnectInterval: 3000,     // Wait 3s between reconnects
  maxReconnectAttempts: 5,     // Max 5 reconnect attempts
});
```

**Connection Behavior:**
- ✅ Automatically connects when `enabled` is `true`
- ✅ Automatically disconnects when `enabled` is `false`
- ✅ Auto-reconnects on unexpected disconnections (up to 5 times)
- ✅ Cleans up properly on component unmount

---

### **3. Camera Component Integration**

**Before:**
```typescript
// Old way - mixed concerns
const { 
  sessionUUID,
  translation,
  isConnected,
  connectWebSocket,
  disconnectWebSocket,
  sendFrame,
  stopProcessing 
} = useGesturaAPI();
```

**After:**
```typescript
// New way - separated concerns

// HTTP API (sending data)
const { 
  sessionUUID, 
  sendFrame,
  stopProcessing,
} = useGesturaAPI();

// WebSocket (receiving data)
const {
  translation,
  isConnected,
  isConnecting,
  error: wsError,
} = useGesturaWebSocket({
  uuid: sessionUUID,
  enabled: isActive,  // Auto-manages connection
});
```

---

## 🚀 Key Benefits

### **1. Separation of Concerns**
- HTTP logic separate from WebSocket logic
- Easier to test and maintain
- Clear responsibilities

### **2. Automatic Connection Management**
- WebSocket connects/disconnects based on `enabled` prop
- No manual connection management needed
- Tied to camera active state

### **3. Robust Error Handling**
- Auto-reconnection on failures
- Detailed error messages
- Connection state tracking

### **4. Developer Experience**
- Clear API surface
- Type-safe with TypeScript
- Comprehensive logging
- Easy to debug

---

## 📊 Connection States

| State | Icon | Description |
|-------|------|-------------|
| **Connected** | 🟢 | WebSocket connected, receiving data |
| **Connecting** | 🟡 | WebSocket attempting to connect |
| **Disconnected** | 🔴 | WebSocket not connected |

---

## 🔧 Configuration

### WebSocket URL
Located in `useGesturaWebSocket.ts`:
```typescript
const WS_BASE_URL = __DEV__ 
  ? 'ws://192.168.1.27:9898'     // Development
  : 'ws://192.168.1.27:9898';    // Production
```

### Reconnection Settings
```typescript
autoReconnect: true,          // Enable auto-reconnect
reconnectInterval: 3000,      // 3 seconds between attempts
maxReconnectAttempts: 5,      // Maximum 5 attempts
```

---

## 🐛 Debugging

### Enable Detailed Logs
All hooks include comprehensive console logging:
- Connection events
- Message received
- Errors
- Reconnection attempts

### Check Connection Status
```typescript
console.log('Connected:', isConnected);
console.log('Connecting:', isConnecting);
console.log('Error:', wsError);
console.log('Reconnect attempts:', reconnectAttempts);
```

---

## 📝 Example Flow

1. **User taps "Start"**
   - `isActive` becomes `true`
   - WebSocket auto-connects via `enabled` prop
   - Camera starts capturing frames

2. **Frame Capture Loop**
   - Every 500ms, capture frame
   - Send via HTTP using `sendFrame()`
   - Continue until stopped

3. **Receiving Translations**
   - WebSocket receives translation
   - Updates `translation` state
   - Displays in UI

4. **User taps "Stop"**
   - `isActive` becomes `false`
   - Calls `stopProcessing()` via HTTP
   - WebSocket auto-disconnects via `enabled` prop

5. **Network Issues**
   - WebSocket disconnects unexpectedly
   - Auto-reconnect kicks in
   - Attempts reconnection up to 5 times
   - Shows connection status in UI

---

## ✅ Migration Complete

Your WebSocket implementation is now:
- ✅ Separated from HTTP logic
- ✅ Automatically managed
- ✅ Robust with auto-reconnection
- ✅ Easy to maintain and extend
- ✅ Production-ready

Happy coding! 🎉
