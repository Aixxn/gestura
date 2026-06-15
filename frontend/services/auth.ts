import axios from "axios";

import apiClient from "./api";

export interface RegisterInput {
  email: string;
  full_name: string;
  password: string;
}

export interface LoginInput {
  email: string;
  password: string;
}

export async function registerUser(input: RegisterInput) {
  const email = input.email.trim().toLowerCase();
  const full_name = input.full_name.trim();
  const password = input.password;

  if (!email || !full_name || !password) {
    throw new Error('Please fill in all required fields.');
  }

  try {
    const response = await apiClient.post("/api/auth/register", {
      username: email,
      email,
      full_name,
      password,
    });

    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error) && error.response?.status === 400) {
      throw new Error("Email already registered.");
    }

    throw new Error("Registration failed. Check your connection and try again.");
  }
}

export async function loginUser(input: LoginInput) {
  const email = input.email.trim().toLowerCase();
  const password = input.password;

  if (!email || !password) {
    throw new Error('Please enter your email and password.');
  }

  try {
    const response = await apiClient.post("/api/auth/login", {
      username: email,
      password,
    });

    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error) && error.response?.status === 401) {
      throw new Error("Invalid email or password.");
    }

    throw new Error("Login failed. Check your connection and try again.");
  }
}
