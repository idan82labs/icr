import { AuthToken, User } from '../types/shared';
import { validateToken } from './validator';

/**
 * Validates the authentication token and returns user info.
 * @param token - The authentication token to validate
 * @returns User object if valid, null otherwise
 */
export async function handleAuth(token: AuthToken): Promise<User | null> {
  if (!validateToken(token)) {
    return null;
  }

  // Verify token expiration
  if (token.expiresAt < Date.now()) {
    console.warn('Token expired');
    return null;
  }

  // Fetch user from database
  const user = await fetchUserById(token.userId);
  return user;
}

export async function refreshToken(token: AuthToken): Promise<AuthToken | null> {
  const user = await handleAuth(token);
  if (!user) {
    return null;
  }

  return generateNewToken(user);
}

async function fetchUserById(userId: string): Promise<User | null> {
  // Database lookup logic
  return null;
}

function generateNewToken(user: User): AuthToken {
  return {
    userId: user.id,
    token: crypto.randomUUID(),
    expiresAt: Date.now() + 3600000,
  };
}
