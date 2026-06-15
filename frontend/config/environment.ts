
import Constants from 'expo-constants';

const extra = Constants.expoConfig?.extra || {};

interface EnvironmentConfig {
  API_URL: string;
  WS_URL: string;
  isDevelopment: boolean;
}

const DEVELOPMENT_CONFIG: EnvironmentConfig = {
  API_URL: 'http://192.168.100.5:8080', // Your local Docker IP
  WS_URL: 'ws://192.168.100.5:9898',  
  isDevelopment: true,
};

// Production: Your deployed Docker backend
const PRODUCTION_CONFIG: EnvironmentConfig = {
  API_URL: extra.apiUrl || 'http://YOUR_SERVER_IP:8080', // Replace with your server IP/domain
  WS_URL: extra.wsUrl || 'ws://YOUR_SERVER_IP:9898',     // Replace with your server IP/domain
  isDevelopment: false,
};

export const ENV: EnvironmentConfig = __DEV__ 
  ? DEVELOPMENT_CONFIG 
  : PRODUCTION_CONFIG;

export const API_BASE_URL = ENV.API_URL;
export const WS_BASE_URL = ENV.WS_URL;
export const logEnvironment = () => {
  console.log('=== Environment Configuration ===');
  console.log('Mode:', ENV.isDevelopment ? 'DEVELOPMENT' : 'PRODUCTION');
  console.log('API URL:', ENV.API_URL);
  console.log('WebSocket URL:', ENV.WS_URL);
  console.log('================================');
};
