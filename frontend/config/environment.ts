/**
 * Environment Configuration
 * 
 * This file manages API and WebSocket URLs for different environments.
 * Update these values based on where your Docker backend is deployed.
 */

import Constants from 'expo-constants';

// Get configuration from app.json extra field or use defaults
const extra = Constants.expoConfig?.extra || {};

interface EnvironmentConfig {
  API_URL: string;
  WS_URL: string;
  isDevelopment: boolean;
}

// Development: Local network (for testing during development)
const DEVELOPMENT_CONFIG: EnvironmentConfig = {
  API_URL: 'http://192.168.1.29:8080', // Your local Docker IP
  WS_URL: 'ws://192.168.1.29:9898',  
  isDevelopment: true,
};

// Production: Your deployed Docker backend
// Update these URLs to match your production server
const PRODUCTION_CONFIG: EnvironmentConfig = {
  API_URL: extra.apiUrl || 'http://YOUR_SERVER_IP:8080', // Replace with your server IP/domain
  WS_URL: extra.wsUrl || 'ws://YOUR_SERVER_IP:9898',     // Replace with your server IP/domain
  isDevelopment: false,
};

// Select configuration based on environment
export const ENV: EnvironmentConfig = __DEV__ 
  ? DEVELOPMENT_CONFIG 
  : PRODUCTION_CONFIG;

// Export individual values for convenience
export const API_BASE_URL = ENV.API_URL;
export const WS_BASE_URL = ENV.WS_URL;

// Logging helper
export const logEnvironment = () => {
  console.log('=== Environment Configuration ===');
  console.log('Mode:', ENV.isDevelopment ? 'DEVELOPMENT' : 'PRODUCTION');
  console.log('API URL:', ENV.API_URL);
  console.log('WebSocket URL:', ENV.WS_URL);
  console.log('================================');
};
