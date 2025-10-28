/**
 * Gestura API Hook
 * 
 * Manages HTTP API calls to Gestura backend
 * For WebSocket connections, use useGesturaWebSocket hook
 */

import { useEffect, useState, useCallback } from 'react';
import { gesturaAPI } from '../services/api';
import { generateUUID } from '../utils/helpers';

export const useGesturaAPI = () => {
  const [sessionUUID, setSessionUUID] = useState<string>('');
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Initialize session UUID on mount
  useEffect(() => {
    const uuid = generateUUID();
    setSessionUUID(uuid);
    console.log('API: Session UUID initialized:', uuid);
  }, []);

  // Send frame to API Gateway
  const sendFrame = useCallback(async (imageData: string | Blob) => {
    if (!sessionUUID) {
      console.error('API: No session UUID available');
      setError('Session not initialized');
      return { success: false, error: 'Session not initialized' };
    }

    if (isSending) {
      console.warn('API: Previous frame still sending, skipping...');
      return { success: false, error: 'Previous request in progress' };
    }

    setIsSending(true);
    setError(null);

    try {
      const response = await gesturaAPI.convertImage(sessionUUID, imageData);
      console.log('API: Frame sent successfully:', response.data);
      setIsSending(false);
      return { success: true, data: response.data };
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to send frame';
      console.error('API: Failed to send frame:', errorMessage);
      setError(errorMessage);
      setIsSending(false);
      return { success: false, error: errorMessage };
    }
  }, [sessionUUID, isSending]);

  // Stop processing for the current session
  const stopProcessing = useCallback(async () => {
    if (!sessionUUID) {
      console.error('API: No session UUID available');
      return { success: false, error: 'Session not initialized' };
    }

    try {
      await gesturaAPI.stopProcessing(sessionUUID);
      console.log('API: Processing stopped successfully');
      return { success: true };
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to stop processing';
      console.error('API: Failed to stop processing:', errorMessage);
      setError(errorMessage);
      return { success: false, error: errorMessage };
    }
  }, [sessionUUID]);

  // Health check
  const checkHealth = useCallback(async () => {
    try {
      const response = await gesturaAPI.healthCheck();
      console.log('API: Health check passed:', response.data);
      return { success: true, data: response.data };
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Health check failed';
      console.error('API: Health check failed:', errorMessage);
      return { success: false, error: errorMessage };
    }
  }, []);

  return {
    sessionUUID,
    isSending,
    error,
    sendFrame,
    stopProcessing,
    checkHealth,
  };
};
