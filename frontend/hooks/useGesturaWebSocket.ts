

import { useEffect, useRef, useState, useCallback } from 'react';
import { WS_BASE_URL } from '../config/environment';

interface UseGesturaWebSocketProps {
  uuid: string;
  enabled?: boolean; // Control when to connect
  autoReconnect?: boolean; // Auto-reconnect on disconnect
  reconnectInterval?: number; // Delay between reconnect attempts (ms)
  maxReconnectAttempts?: number; // Max number of reconnect attempts
}

interface WebSocketState {
  isConnected: boolean;
  isConnecting: boolean;
  error: string | null;
  reconnectAttempts: number;
}

export const useGesturaWebSocket = ({
  uuid,
  enabled = true,
  autoReconnect = true,
  reconnectInterval = 3000,
  maxReconnectAttempts = 5,
}: UseGesturaWebSocketProps) => {
  const [translation, setTranslation] = useState<string>('');
  const [state, setState] = useState<WebSocketState>({
    isConnected: false,
    isConnecting: false,
    error: null,
    reconnectAttempts: 0,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const shouldConnectRef = useRef(enabled);

  // Update the shouldConnect ref when enabled changes
  useEffect(() => {
    shouldConnectRef.current = enabled;
  }, [enabled]);

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (!uuid || !shouldConnectRef.current) {
      console.log('WebSocket: Connection skipped (no UUID or not enabled)');
      return;
    }

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      console.log('WebSocket: Already connected');
      return;
    }

    if (wsRef.current?.readyState === WebSocket.CONNECTING) {
      console.log('WebSocket: Already connecting');
      return;
    }

    try {
      const wsUrl = `${WS_BASE_URL}?uuid=${uuid}`;
      console.log(`WebSocket: Connecting to ${wsUrl}`);

      setState(prev => ({ ...prev, isConnecting: true, error: null }));

      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log('WebSocket: Connected successfully');
        setState({
          isConnected: true,
          isConnecting: false,
          error: null,
          reconnectAttempts: 0,
        });

        // Clear any pending reconnect timeouts
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
          reconnectTimeoutRef.current = null;
        }
      };

      ws.onmessage = (event) => {
        console.log('WebSocket: Translation received:', event.data);
        setTranslation(event.data);
      };

      ws.onerror = (error) => {
        console.error('WebSocket: Error occurred:', error);
        setState(prev => ({
          ...prev,
          error: 'WebSocket connection error',
          isConnecting: false,
        }));
      };

      ws.onclose = (event) => {
        console.log(`WebSocket: Disconnected (code: ${event.code}, reason: ${event.reason})`);
        
        setState(prev => ({
          ...prev,
          isConnected: false,
          isConnecting: false,
        }));

        wsRef.current = null;

        // Attempt to reconnect if enabled and within max attempts
        if (
          autoReconnect && 
          shouldConnectRef.current && 
          state.reconnectAttempts < maxReconnectAttempts
        ) {
          const nextAttempt = state.reconnectAttempts + 1;
          console.log(`WebSocket: Reconnecting (attempt ${nextAttempt}/${maxReconnectAttempts}) in ${reconnectInterval}ms...`);
          
          setState(prev => ({
            ...prev,
            reconnectAttempts: nextAttempt,
          }));

          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
        } else if (state.reconnectAttempts >= maxReconnectAttempts) {
          console.error('WebSocket: Max reconnection attempts reached');
          setState(prev => ({
            ...prev,
            error: 'Failed to reconnect after maximum attempts',
          }));
        }
      };

      wsRef.current = ws;
    } catch (error) {
      console.error('WebSocket: Failed to create connection:', error);
      setState(prev => ({
        ...prev,
        isConnecting: false,
        error: error instanceof Error ? error.message : 'Connection failed',
      }));
    }
  }, [uuid, autoReconnect, reconnectInterval, maxReconnectAttempts, state.reconnectAttempts]);

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    console.log('WebSocket: Manually disconnecting...');
    shouldConnectRef.current = false;

    // Clear any pending reconnect timeouts
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    // Close the WebSocket connection
    if (wsRef.current) {
      wsRef.current.close(1000, 'Client disconnect');
      wsRef.current = null;
    }

    // Reset state
    setState({
      isConnected: false,
      isConnecting: false,
      error: null,
      reconnectAttempts: 0,
    });

    setTranslation('');
  }, []);

  // Manual reconnect
  const reconnect = useCallback(() => {
    console.log('WebSocket: Manual reconnect triggered');
    disconnect();
    setTimeout(() => {
      shouldConnectRef.current = true;
      connect();
    }, 500);
  }, [disconnect, connect]);

  // Clear translation
  const clearTranslation = useCallback(() => {
    setTranslation('');
  }, []);

  // Connect/disconnect based on enabled prop
  useEffect(() => {
    if (enabled && uuid) {
      connect();
    } else {
      disconnect();
    }

    // Cleanup on unmount
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmount');
      }
    };
  }, [enabled, uuid, connect, disconnect]);

  return {
    // State
    translation,
    isConnected: state.isConnected,
    isConnecting: state.isConnecting,
    error: state.error,
    reconnectAttempts: state.reconnectAttempts,
    
    // Actions
    connect,
    disconnect,
    reconnect,
    clearTranslation,
  };
};
