// In-memory token storage
// For production, replace with expo-secure-store or AsyncStorage
let accessToken: string | null = null;

export function setToken(token: string | null) {
  accessToken = token;
}

export function getToken(): string | null {
  return accessToken;
}

export function clearToken() {
  accessToken = null;
}
