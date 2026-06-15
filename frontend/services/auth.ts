import axios from 'axios';

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

function getAuthError(error: unknown, fallback: string): Error {
  if (axios.isAxiosError(error)) {
    const message = error.response?.data?.message;
    if (typeof message === 'string' && message.length > 0) {
      return new Error(message);
    }
  }

  return new Error(fallback);
}

export async function registerUser(input: RegisterInput) {
  const email = input.email.trim().toLowerCase();
  const full_name = input.full_name.trim();
  const password = input.password;

  if (!email || !full_name || !password) {
    throw new Error('Please fill in all required fields.');
  }

  try {
    const response = await apiClient.post<AuthResponse>('/api/auth/register', {
      email,
      password,
      full_name,
    });

    const data = response.data;

    if (data.token) {
      setToken(data.token);
    }

    return data;
  } catch (error) {
    throw getAuthError(error, 'Registration failed. Check your connection and try again.');
  }
}

export async function loginUser(input: LoginInput) {
  const email = input.email.trim().toLowerCase();
  const password = input.password;

  if (!email || !password) {
    throw new Error('Please enter your email and password.');
  }

  try {
    const response = await apiClient.post<AuthResponse>('/api/auth/login', {
      email,
      password,
    });

    const data = response.data;

    if (data.token) {
      setToken(data.token);
    }

    return data;
  } catch (error) {
    throw getAuthError(error, 'Login failed. Check your connection and try again.');
  }
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
