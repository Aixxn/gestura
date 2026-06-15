import { useEffect, useRef, useState, useCallback } from 'react';
import { createWebSocketConnection } from '../services/api';

interface UseGesturaWebSocketProps {
  uuid: string;
  enabled?: boolean;
  autoReconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
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
  const [aslGloss, setAslGloss] = useState<string>('');
  const [state, setState] = useState<WebSocketState>({
    isConnected: false,
    isConnecting: false,
    error: null,
    reconnectAttempts: 0,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const shouldConnectRef = useRef(enabled);
  const reconnectAttemptsRef = useRef(0);

  useEffect(() => {
    shouldConnectRef.current = enabled;
  }, [enabled]);

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
      console.log(`WebSocket: Connecting for UUID ${uuid}`);

      setState(prev => ({ ...prev, isConnecting: true, error: null }));

      const ws = createWebSocketConnection(uuid);
      if (!ws) {
        console.error('WebSocket: Failed to create connection (no token?)');
        setState(prev => ({
          ...prev,
          isConnecting: false,
          error: 'Authentication token not available',
        }));
        return;
      }

      ws.onopen = () => {
        console.log('WebSocket: Connected successfully');
        reconnectAttemptsRef.current = 0;
        setState({
          isConnected: true,
          isConnecting: false,
          error: null,
          reconnectAttempts: 0,
        });

        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
          reconnectTimeoutRef.current = null;
        }
      };

      ws.onmessage = (event) => {
        try {
          const raw = event.data;
          const parsed = typeof raw === 'string' ? JSON.parse(raw) : raw;
          console.log('WebSocket: Raw message received:', parsed);

          const payloadUuid = parsed?.uuid;

          if (payloadUuid && payloadUuid !== uuid) {
            console.log('WebSocket: Ignoring message for different UUID:', payloadUuid);
            return;
          }

          const english = parsed?.english;
          if (!english) {
            console.log('WebSocket: No english field in message — skipping');
            return;
          }

          setTranslation(String(english));

          if (parsed?.asl_gloss) {
            setAslGloss(String(parsed.asl_gloss));
          }
        } catch (err) {
          console.error('WebSocket: Failed to parse incoming message:', err, event.data);
        }
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

        if (
          autoReconnect &&
          shouldConnectRef.current &&
          reconnectAttemptsRef.current < maxReconnectAttempts
        ) {
          reconnectAttemptsRef.current += 1;
          const attempt = reconnectAttemptsRef.current;
          console.log(`WebSocket: Reconnecting (attempt ${attempt}/${maxReconnectAttempts}) in ${reconnectInterval}ms...`);

          setState(prev => ({
            ...prev,
            reconnectAttempts: attempt,
          }));

          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
        } else if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
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
  }, [uuid, autoReconnect, reconnectInterval, maxReconnectAttempts]);

  const disconnect = useCallback(() => {
    console.log('WebSocket: Manually disconnecting...');
    shouldConnectRef.current = false;

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close(1000, 'Client disconnect');
      wsRef.current = null;
    }

    reconnectAttemptsRef.current = 0;
    setState({
      isConnected: false,
      isConnecting: false,
      error: null,
      reconnectAttempts: 0,
    });

    setTranslation('');
    setAslGloss('');
  }, []);

  const reconnect = useCallback(() => {
    console.log('WebSocket: Manual reconnect triggered');
    disconnect();
    setTimeout(() => {
      shouldConnectRef.current = true;
      connect();
    }, 500);
  }, [disconnect, connect]);

  const clearTranslation = useCallback(() => {
    setTranslation('');
    setAslGloss('');
  }, []);

  useEffect(() => {
    if (enabled && uuid) {
      connect();
    } else {
      disconnect();
    }

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
    translation,
    aslGloss,
    isConnected: state.isConnected,
    isConnecting: state.isConnecting,
    error: state.error,
    reconnectAttempts: state.reconnectAttempts,
    connect,
    disconnect,
    reconnect,
    clearTranslation,
  };
};
