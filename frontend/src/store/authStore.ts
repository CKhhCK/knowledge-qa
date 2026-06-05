/**
 * Auth store — user login state and JWT token management.
 */

import { create } from 'zustand';

const TOKEN_KEY = 'helloagents_qa_token';
const USER_KEY = 'helloagents_qa_user';

interface User {
  user_id: string;
  username: string;
  email?: string;
  created_at?: string;
}

interface AuthState {
  token: string | null;
  user: User | null;
  isLoggedIn: boolean;

  setAuth: (token: string, user: User) => void;
  logout: () => void;
  loadFromStorage: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  token: localStorage.getItem(TOKEN_KEY),
  user: JSON.parse(localStorage.getItem(USER_KEY) || 'null'),
  isLoggedIn: !!localStorage.getItem(TOKEN_KEY),

  setAuth: (token, user) => {
    localStorage.setItem(TOKEN_KEY, token);
    localStorage.setItem(USER_KEY, JSON.stringify(user));
    set({ token, user, isLoggedIn: true });
  },

  logout: () => {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    set({ token: null, user: null, isLoggedIn: false });
  },

  loadFromStorage: () => {
    const token = localStorage.getItem(TOKEN_KEY);
    const user = JSON.parse(localStorage.getItem(USER_KEY) || 'null');
    set({ token, user, isLoggedIn: !!token });
  },
}));

/** Get the stored auth token for API calls */
export function getAuthToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}
