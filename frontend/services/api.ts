import axios from 'axios';
import { API_BASE_URL, WS_BASE_URL } from '../config/environment';
import { getToken } from './token';

export interface ConvertImageResponse {
  success: boolean;
  message?: string;
  data?: any;
}

export interface StopProcessingResponse {
  success: boolean;
  message: string;
}

export interface HealthCheckResponse {
  status: string;
  timestamp: number;
}

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor: attach auth token
apiClient.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.baseURL}${config.url}`);

    const token = getToken();
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor: handle 401 globally
apiClient.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.config.url} - ${response.status}`);
    return response;
  },
  (error) => {
    if (error.response?.status === 401) {
      console.warn('API: Unauthorized - token may be expired');
    }
    console.error('API Response Error:', error.response?.status, error.message);
    return Promise.reject(error);
  }
);

export const gesturaAPI = {
  convertImage: async (uuid: string, imageFile: Blob | File | string) => {
    const formData = new FormData();
    formData.append('uuid', uuid);

    if (typeof imageFile === 'string') {
      const uri = imageFile.startsWith('file://') ? imageFile : `file://${imageFile}`;
      formData.append('rawImage', {
        uri: uri,
        type: 'image/jpeg',
        name: 'frame.jpg',
      } as any);
    } else {
      formData.append('rawImage', {
        uri: imageFile,
        type: 'image/jpeg',
        name: 'frame.jpg',
      } as any);
    }

    try {
      const token = getToken();
      const headers: Record<string, string> = {};
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }

      const response = await fetch(`${API_BASE_URL}/api/convert`, {
        method: 'POST',
        body: formData,
        headers,
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

  stopProcessing: async (uuid: string) => {
    return apiClient.get(`/api/stop/${uuid}`);
  },

  healthCheck: async () => {
    return apiClient.get('/health');
  },
};

export const createWebSocketConnection = (uuid: string): WebSocket | null => {
  const token = getToken();
  if (!token) {
    console.error('WebSocket: No auth token available');
    return null;
  }

  const WS_URL = `${WS_BASE_URL}?uuid=${uuid}&token=${encodeURIComponent(token)}`;
  console.log(`Connecting to WebSocket: ${WS_URL}`);
  return new WebSocket(WS_URL);
};

export default apiClient;
