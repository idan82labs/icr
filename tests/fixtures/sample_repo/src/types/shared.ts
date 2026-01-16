/**
 * Shared type definitions for the authentication system.
 */

export interface AuthToken {
  token: string;
  userId: string;
  expiresAt: number;
}

export interface User {
  id: string;
  email: string;
  name: string;
  roles: string[];
  createdAt: number;
  lastLogin: number;
}

export interface ApiResponse<T> {
  success: boolean;
  data: T | null;
  error?: string;
  metadata?: Record<string, unknown>;
}

export type AuthRole = 'admin' | 'user' | 'guest';

export interface Permission {
  resource: string;
  action: 'read' | 'write' | 'delete' | 'admin';
}
