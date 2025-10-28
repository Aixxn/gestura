import axios from 'axios';
import { API_BASE_URL, logEnvironment } from '../config/environment';

// Log environment configuration on import
logEnvironment();

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
apiClient.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.baseURL}${config.url}`);
    if (config.headers) {
      console.log('Request Headers:', JSON.stringify(config.headers, null, 2));
    }
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for logging
apiClient.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.config.url} - ${response.status}`);
    return response;
  },
  (error) => {
    console.error('API Response Error:', error.response?.status, error.message);
    return Promise.reject(error);
  }
);

// API Service Methods
export const gesturaAPI = {
  /**
   * Convert image to sign language translation
   * @param uuid - Unique session identifier
   * @param imageFile - Image file path or blob
   * @returns Promise with response
   */
  convertImage: async (uuid: string, imageFile: Blob | File | string) => {
    const formData = new FormData();
    formData.append('uuid', uuid);
    
    // Handle different input types
    if (typeof imageFile === 'string') {
      // If it's a file path, create proper file object for React Native
      const uri = imageFile.startsWith('file://') ? imageFile : `file://${imageFile}`;
      formData.append('rawImage', {
        uri: uri,
        type: 'image/jpeg',
        name: 'frame.jpg',
      } as any);
    } else {
      // If it's a Blob or File
      formData.append('rawImage', {
        uri: imageFile,
        type: 'image/jpeg',
        name: 'frame.jpg',
      } as any);
    }

    // Use fetch instead of axios for better React Native FormData support
    // Don't set Content-Type header - let fetch set it with boundary
    try {
      const response = await fetch(`${API_BASE_URL}/api/convert`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`HTTP error! status: ${response.status}, body: ${errorText}`);
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Upload successful:', data);
      return { data };
    } catch (error) {
      console.error('Fetch error:', error);
      throw error;
    }
  },

  /**
   * Stop processing for a session
   * @param uuid - Unique session identifier
   * @returns Promise with response
   */
  stopProcessing: async (uuid: string) => {
    return apiClient.post('/api/stop', { uuid });
  },

  /**
   * Health check
   * @returns Promise with response
   */
  healthCheck: async () => {
    return apiClient.get('/health');
  },
};

// WebSocket connection helper
export const createWebSocketConnection = (uuid: string): WebSocket => {
  // Import WS_BASE_URL from environment config
  const { WS_BASE_URL } = require('../config/environment');
  const WS_URL = `${WS_BASE_URL}?uuid=${uuid}`;

  console.log(`Connecting to WebSocket: ${WS_URL}`);
  return new WebSocket(WS_URL);
};

export default apiClient;
