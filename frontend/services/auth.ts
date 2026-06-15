import apiClient from './api';
import { setToken, clearToken } from './token';

export interface RegisterInput {
  email: string;
  full_name: string;
  password: string;
}

export interface LoginInput {
  email: string;
  password: string;
}

export interface AuthResponse {
  success: boolean;
  message: string;
  token?: string;
  user?: {
    _id?: string;
    email: string;
    full_name?: string;
  };
}

export async function registerUser(input: RegisterInput) {
  const response = await apiClient.post<AuthResponse>('/api/auth/register', {
    email: input.email,
    password: input.password,
    full_name: input.full_name,
  });

  const data = response.data;

  if (data.token) {
    setToken(data.token);
  }

  return data;
}

export async function loginUser(input: LoginInput) {
  const response = await apiClient.post<AuthResponse>('/api/auth/login', {
    email: input.email,
    password: input.password,
  });

  const data = response.data;

  if (data.token) {
    setToken(data.token);
  }

  return data;
}

export async function logoutUser() {
  try {
    await apiClient.post('/api/auth/logout');
  } finally {
    clearToken();
  }
}

export async function getCurrentUser() {
  const response = await apiClient.get('/api/auth/me');
  return response.data;
}
