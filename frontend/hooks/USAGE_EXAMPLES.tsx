/**
 * EXAMPLE: Using the separated WebSocket and API hooks
 * 
 * This example shows how to use both hooks together in a component
 */

import { useGesturaAPI } from './useGesturaAPI';
import { useGesturaWebSocket } from './useGesturaWebSocket';
import { useState } from 'react';

export function ExampleUsage() {
  const [isActive, setIsActive] = useState(false);

  // 1. Initialize HTTP API (for sending frames)
  const { 
    sessionUUID,      // UUID for this session
    sendFrame,        // Function to send frames
    stopProcessing,   // Function to stop processing
    isSending,        // Is a frame currently being sent?
    error: apiError   // Any API errors
  } = useGesturaAPI();

  // 2. Initialize WebSocket (for receiving translations)
  const {
    translation,        // Latest translation received
    isConnected,        // Is WebSocket connected?
    isConnecting,       // Is WebSocket currently connecting?
    error: wsError,     // Any WebSocket errors
    reconnectAttempts,  // Number of reconnect attempts made
    reconnect,          // Manually trigger reconnection
    clearTranslation,   // Clear the translation
  } = useGesturaWebSocket({
    uuid: sessionUUID,           // Use the same UUID from API hook
    enabled: isActive,           // Only connect when active
    autoReconnect: true,         // Auto-reconnect on disconnect
    reconnectInterval: 3000,     // Wait 3s between reconnects
    maxReconnectAttempts: 5,     // Max 5 reconnect attempts
  });

  // 3. Handle starting/stopping
  const handleToggle = async () => {
    if (isActive) {
      // Stopping
      await stopProcessing();
      setIsActive(false); // WebSocket will auto-disconnect
      clearTranslation();
    } else {
      // Starting
      setIsActive(true); // WebSocket will auto-connect
    }
  };

  // 4. Send a frame (example)
  const handleSendFrame = async (frameData: string) => {
    const result = await sendFrame(frameData);
    if (result.success) {
      console.log('Frame sent successfully!');
    } else {
      console.error('Failed to send frame:', result.error);
    }
  };

  return (
    <div>
      <h1>Gestura Example</h1>
      
      {/* Connection Status */}
      <div>
        <p>Session: {sessionUUID}</p>
        <p>
          WebSocket: {
            isConnected ? 'Connected' : 
            isConnecting ? '🟡Connecting...' : 
            '🔴 Disconnected'
          }
        </p>
        {reconnectAttempts > 0 && (
          <p>Reconnect attempts: {reconnectAttempts}/5</p>
        )}
      </div>

      {/* Errors */}
      {apiError && <p style={{ color: 'red' }}>API Error: {apiError}</p>}
      {wsError && <p style={{ color: 'red' }}>WebSocket Error: {wsError}</p>}

      {/* Translation Display */}
      <div>
        <h2>Translation</h2>
        <p>{translation || 'Waiting for translation...'}</p>
      </div>

      {/* Controls */}
      <button onClick={handleToggle}>
        {isActive ? 'Stop' : 'Start'}
      </button>
      
      {!isConnected && reconnectAttempts >= 5 && (
        <button onClick={reconnect}>
          Retry Connection
        </button>
      )}
    </div>
  );
}

/**
 * EXAMPLE: Custom configuration
 */
export function ExampleWithCustomConfig() {
  const [isActive, setIsActive] = useState(false);
  
  const { sessionUUID } = useGesturaAPI();
  
  const {
    translation,
    isConnected,
    error,
  } = useGesturaWebSocket({
    uuid: sessionUUID,
    enabled: isActive,
    autoReconnect: true,
    reconnectInterval: 5000,      // Custom: Wait 5s between reconnects
    maxReconnectAttempts: 10,     // Custom: Try 10 times before giving up
  });

  return (
    <div>
      <h1>Custom Configuration Example</h1>
      <p>Reconnect Interval: 5 seconds</p>
      <p>Max Attempts: 10</p>
      <button onClick={() => setIsActive(!isActive)}>
        {isActive ? 'Stop' : 'Start'}
      </button>
    </div>
  );
}

/**
 * EXAMPLE: Manual connection control
 */
export function ExampleWithManualControl() {
  const { sessionUUID } = useGesturaAPI();
  
  const {
    translation,
    isConnected,
    connect,
    disconnect,
    reconnect,
  } = useGesturaWebSocket({
    uuid: sessionUUID,
    enabled: false,  // Don't auto-connect
    autoReconnect: false,  // Don't auto-reconnect
  });

  return (
    <div>
      <h1>Manual Control Example</h1>
      <p>Status: {isConnected ? 'Connected' : 'Disconnected'}</p>
      <p>Translation: {translation}</p>
      
      <button onClick={() => connect()}>Connect</button>
      <button onClick={() => disconnect()}>Disconnect</button>
      <button onClick={() => reconnect()}>Reconnect</button>
    </div>
  );
}
