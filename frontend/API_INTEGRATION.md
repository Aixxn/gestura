# Gestura API Integration

## Overview

The Gestura API client provides easy access to the API Gateway for sign language translation.

## Setup

### 1. Configuration

The API is configured in `services/api.ts`:

- **Development (Android Emulator)**: `http://10.0.2.2:8080`
- **Production (Real Device)**: Update with your computer's IP address

### 2. WebSocket Configuration

WebSocket endpoint: `ws://10.0.2.2:9898` (emulator) or `ws://<YOUR_IP>:9898` (real device)

## Usage

### Basic Example

```typescript
import { useGesturaAPI } from '../hooks/useGesturaAPI';

function MyComponent() {
  const { 
    sessionUUID, 
    translation, 
    isConnected,
    connectWebSocket,
    sendFrame,
    stopProcessing 
  } = useGesturaAPI();

  // Connect WebSocket when component mounts
  useEffect(() => {
    connectWebSocket();
  }, []);

  // Send a frame
  const handleSendFrame = async (imageBase64: string) => {
    await sendFrame(imageBase64);
  };

  // Stop processing
  const handleStop = async () => {
    await stopProcessing();
  };

  return (
    <View>
      <Text>Session: {sessionUUID}</Text>
      <Text>Connected: {isConnected ? 'Yes' : 'No'}</Text>
      <Text>Translation: {translation}</Text>
    </View>
  );
}
```

### Integration with Camera Component

```typescript
import { Camera } from 'react-native-vision-camera';
import { useGesturaAPI } from '../hooks/useGesturaAPI';

export default function CameraComponent() {
  const { 
    sessionUUID,
    translation,
    isConnected,
    connectWebSocket,
    disconnectWebSocket,
    sendFrame,
    stopProcessing 
  } = useGesturaAPI();

  const [isActive, setIsActive] = useState(false);

  // Connect WebSocket when starting
  const handleStart = () => {
    setIsActive(true);
    connectWebSocket();
  };

  // Disconnect and stop when stopping
  const handleStop = async () => {
    setIsActive(false);
    await stopProcessing();
    disconnectWebSocket();
  };

  // Capture and send frames every 500ms
  useEffect(() => {
    if (!isActive) return;

    const interval = setInterval(async () => {
      try {
        // Take photo from camera
        const photo = await camera.current?.takePhoto({
          qualityPrioritization: 'speed',
          enableShutterSound: false,
        });

        if (photo) {
          // Send frame to API
          await sendFrame(`file://${photo.path}`);
        }
      } catch (error) {
        console.error('Failed to capture/send frame:', error);
      }
    }, 500);

    return () => clearInterval(interval);
  }, [isActive, sendFrame]);

  return (
    <View>
      <Camera ref={camera} />
      <Text>Translation: {translation || 'Waiting...'}</Text>
      <Button onPress={handleStart} title="Start" />
      <Button onPress={handleStop} title="Stop" />
    </View>
  );
}
```

## API Methods

### `gesturaAPI.convertImage(uuid: string, imageFile: Blob | File)`

Send an image frame for sign language detection.

**Parameters:**
- `uuid` - Unique session identifier
- `imageFile` - Image blob or file

**Returns:** Promise with response

**Example:**
```typescript
const blob = base64ToBlob(imageBase64);
await gesturaAPI.convertImage(sessionUUID, blob);
```

### `gesturaAPI.stopProcessing(uuid: string)`

Stop processing for the current session.

**Parameters:**
- `uuid` - Unique session identifier

**Returns:** Promise with response

**Example:**
```typescript
await gesturaAPI.stopProcessing(sessionUUID);
```

### `createWebSocketConnection(uuid: string)`

Create a WebSocket connection to receive translations.

**Parameters:**
- `uuid` - Unique session identifier

**Returns:** WebSocket instance

**Example:**
```typescript
const ws = createWebSocketConnection(sessionUUID);

ws.onmessage = (event) => {
  console.log('Translation:', event.data);
};
```

## Helper Functions

### `generateUUID()`

Generate a unique session identifier.

**Returns:** UUID string

**Example:**
```typescript
const sessionId = generateUUID();
```

### `base64ToBlob(base64: string, contentType?: string)`

Convert base64 image to Blob.

**Parameters:**
- `base64` - Base64 encoded image
- `contentType` - MIME type (default: 'image/jpeg')

**Returns:** Blob

**Example:**
```typescript
const blob = base64ToBlob(imageBase64, 'image/jpeg');
```

## Configuration for Real Devices

When running on a real device (not emulator), update the IP addresses in `services/api.ts`:

1. Find your computer's IP address:
   - Windows: `ipconfig`
   - Mac/Linux: `ifconfig` or `ip addr`

2. Update the configuration:
```typescript
const API_BASE_URL = __DEV__ 
  ? 'http://10.0.2.2:8080' // Emulator
  : 'http://192.168.1.XXX:8080'; // Replace with your IP

const WS_URL = __DEV__ 
  ? 'ws://10.0.2.2:9898' // Emulator
  : 'ws://192.168.1.XXX:9898'; // Replace with your IP
```

3. Make sure your computer and phone are on the same WiFi network.

## Troubleshooting

### Connection Refused

- Make sure Docker containers are running: `docker compose ps`
- Check if API Gateway is accessible: `curl http://localhost:8080/health`
- Verify your firewall allows connections on ports 8080 and 9898

### WebSocket Not Connecting

- Check WebSocket server is running in API Gateway logs
- Verify UUID is being sent correctly
- Check network connectivity between device and computer

### Images Not Processing

- Check translationService is running: `docker compose logs translationService`
- Check image format is correct (JPEG/PNG)

## Files Created

- `frontend/services/api.ts` - Axios API client configuration
- `frontend/hooks/useGesturaAPI.ts` - React hook for API integration
- `frontend/utils/helpers.ts` - Utility functions (UUID, base64 conversion)
